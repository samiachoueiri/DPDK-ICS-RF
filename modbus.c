/* SPDX-License-Identifier: BSD-3-Clause
 * Modbus-TCP sensor drift detector (DPDK)
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <stdlib.h>
#include <inttypes.h>
#include <getopt.h>
#include <signal.h>
#include <arpa/inet.h>

#include <rte_memory.h>
#include <rte_launch.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>
#include <rte_debug.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_mbuf.h>
#include <rte_mbuf_dyn.h>
#include <rte_version.h>
#include <rte_common.h>
#include <rte_malloc.h>
#include <rte_mempool.h>
#include <math.h>  // for fabsf, fmaxf, fminf
#include <rte_hash.h>
#include <rte_jhash.h>
#include <rte_spinlock.h>
#include <rte_atomic.h>

#include <arm_neon.h> // NEON for predict_mlp

/* model and normalization (user-provided) */
#include "mlp_weights.h"
#include "feature_stats.h"

/* --- config --- */
#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS (8191 * 2)
#define MBUF_CACHE_SIZE 512
#define BURST_SIZE 32

#define N_PACKETS 6  /* not used; keep for compatibility */
#define WINDOW_SIZE 32 /* number of sensor samples per window */

/* tune this according to memory */
#define MAX_SENSORS_PER_CORE 500000
#define MAX_CORES RTE_MAX_LCORE
#define MAX_PENDING_REQUESTS 10000

#define EWMA_ALPHA 0.3f
#define CUSUM_ALLOWANCE 0.5f
#define CUSUM_THRESHOLD 2.0f
#define MAX_TIMING_SAMPLES_PER_CORE 65536

#define ALIGN16 __attribute__((aligned(16)))

/* --- types --- */

/* Modbus TCP Header (MBAP) */
struct modbus_mbap {
    uint16_t trans_id;
    uint16_t protocol_id;
    uint16_t length;
    uint8_t unit_id;
} __attribute__((packed));

/* Modbus Request PDU */
struct modbus_request_pdu {
    uint8_t function_code;
    uint16_t start_addr;
    uint16_t quantity;
} __attribute__((packed));

/* Modbus Response PDU */
struct modbus_response_pdu {
    uint8_t function_code;
    uint8_t byte_count;
    /* Followed by register data */
} __attribute__((packed));

/* Key identifying a monitored sensor stream */
struct sensor_key {
    uint8_t unit_id;
    uint16_t reg_addr;  /* register address to track individual sensors */
} __attribute__((packed));

/* Enhanced sensor entry with CUSUM/EWMA state */
struct sensor_entry {
    uint64_t timestamps[WINDOW_SIZE];
    int32_t readings[WINDOW_SIZE];
    uint16_t sample_count;
    uint16_t write_index;
    uint8_t finalized;
    uint8_t processed;
    
    /* CUSUM/EWMA state */
    float ewma_value;
    float cusum_positive;
    float cusum_negative;
    float historical_mean;
    uint32_t total_samples;  // For historical mean calculation
};

/* Track pending Modbus requests */
struct pending_request {
    uint16_t trans_id;
    uint8_t unit_id;
    uint16_t start_addr;
    uint16_t num_registers;
    uint64_t timestamp;
};

struct request_tracker {
    struct pending_request *requests;
    uint32_t capacity;
    uint32_t count;
    rte_spinlock_t lock;
};

struct worker_args {
    struct rte_mempool *mbuf_pool;
    struct rte_hash    *sensor_table;
    struct sensor_entry *sensor_pool;
    float *buf_a;
    float *buf_b;
    uint16_t queue_id;
    uint32_t next_free;
    uint16_t port_id;
    uint64_t pred0;
    uint64_t pred1;
    unsigned core_id;
    uint64_t total_readings;  /* counter for sensor readings */

    /* --- timing / sampling buffers added --- */
    uint64_t *feat_cycles;    /* cycles spent computing features (per sample) */
    uint64_t *infer_cycles;   /* cycles spent running MLP (per sample) */
    int32_t  *sample_pred;    /* prediction class per sample */
    size_t    samples_capacity;
    size_t    samples_count;

    uint64_t *burst_cycles;   /* cycles spent processing the whole RX burst */
    size_t    burst_capacity;
    size_t    burst_count;
};


/* per-core arrays */
static struct sensor_entry *sensor_pools[MAX_CORES];
static struct rte_hash *sensor_tables[MAX_CORES];
static struct worker_args worker_args[MAX_CORES];

/* Global request tracker */
static struct request_tracker g_request_tracker;

/* CSV logging and synchronization */
static FILE *g_feat_csv = NULL;
static rte_spinlock_t g_feat_csv_lock = RTE_SPINLOCK_INITIALIZER;

/* atomic counts */
static rte_atomic64_t received_packets;
static rte_atomic64_t processed_packets;

/* shutdown flag */
static volatile sig_atomic_t stop_requested = 0;

/* cached TSC frequency (Hz) */
static double g_tsc_hz = 0.0;

/* forward port init */
static inline int port_init(uint16_t port, struct rte_mempool *mbuf_pool, uint16_t number_rings);

/* --- Request tracking functions --- */

static int init_request_tracker(uint32_t capacity) {
    g_request_tracker.requests = rte_zmalloc("pending_requests", 
        capacity * sizeof(struct pending_request), 0);
    if (!g_request_tracker.requests) return -1;
    
    g_request_tracker.capacity = capacity;
    g_request_tracker.count = 0;
    rte_spinlock_init(&g_request_tracker.lock);
    return 0;
}

static void add_pending_request(uint16_t trans_id, uint8_t unit_id, 
                               uint16_t start_addr, uint16_t num_registers) {
    rte_spinlock_lock(&g_request_tracker.lock);
    
    /* Simple cleanup: remove old requests if near capacity */
    if (g_request_tracker.count >= g_request_tracker.capacity - 100) {
        g_request_tracker.count = g_request_tracker.count / 2; /* Keep half */
    }
    
    if (g_request_tracker.count < g_request_tracker.capacity) {
        struct pending_request *req = &g_request_tracker.requests[g_request_tracker.count];
        req->trans_id = trans_id;
        req->unit_id = unit_id;
        req->start_addr = start_addr;
        req->num_registers = num_registers;
        req->timestamp = rte_rdtsc_precise();
        g_request_tracker.count++;
    }
    
    rte_spinlock_unlock(&g_request_tracker.lock);
}

static int find_pending_request(uint16_t trans_id, uint8_t unit_id, 
                               struct pending_request *result) {
    rte_spinlock_lock(&g_request_tracker.lock);
    
    int found = 0;
    for (uint32_t i = 0; i < g_request_tracker.count; i++) {
        struct pending_request *req = &g_request_tracker.requests[i];
        if (req->trans_id == trans_id && req->unit_id == unit_id) {
            *result = *req;
            /* Remove found request by swapping with last */
            if (i < g_request_tracker.count - 1) {
                *req = g_request_tracker.requests[g_request_tracker.count - 1];
            }
            g_request_tracker.count--;
            found = 1;
            break;
        }
    }
    
    rte_spinlock_unlock(&g_request_tracker.lock);
    return found;
}



/* --- MLP & helpers --- */

/* sigmoid approximation */
static inline float fast_sigmoid(float x) {
    if (x <= -4.0f) return 0.0f;
    else if (x <= -2.0f) return 0.0625f * x + 0.25f;
    else if (x <= 0.0f)  return 0.125f * x + 0.5f;
    else if (x <= 2.0f)  return -0.125f * x + 0.5f;
    else if (x <= 4.0f)  return -0.0625f * x + 0.75f;
    else return 1.0f;
}

/* NEON layer forward */
static void layer_forward_neon(const float *W, const float *B,
                               const float *in, float *out,
                               int size_in, int size_out,
                               int is_output) {
    int j = 0;
    for (; j + 4 <= size_out; j += 4) {
        float32x4_t acc = vld1q_f32(&B[j]);
        for (int k = 0; k < size_in; k++) {
            acc = vfmaq_f32(acc,
                            vdupq_n_f32(in[k]),
                            vld1q_f32(&W[k*size_out + j]));
        }
        if (!is_output)  acc = vmaxq_f32(acc, vdupq_n_f32(0.0f));
        vst1q_f32(&out[j], acc);
    }
    for (; j < size_out; j++) {
        float a = B[j];
        for (int k = 0; k < size_in; k++)
            a += W[k*size_out + j] * in[k];
        out[j] = is_output ? a : (a > 0.0f ? a : 0.0f);
    }
    if (is_output) {
        for (int i = 0; i < size_out; i++)
            out[i] = fast_sigmoid(out[i]);
    }
}

static int predict_mlp(const float *in_features, float *buf_a, float *buf_b) {
    float *in_buf = buf_a, *out_buf = buf_b;
    memcpy(in_buf, in_features, LAYER_SIZES[0] * sizeof(float));

    for (int L = 0; L < NUM_LAYERS; L++) {
        layer_forward_neon(WEIGHTS[L], BIASES[L],
                           in_buf, out_buf,
                           LAYER_SIZES[L],
                           LAYER_SIZES[L+1],
                           (L == NUM_LAYERS - 1));
        float *tmp = in_buf; in_buf = out_buf; out_buf = tmp;
    }

    int final_size = LAYER_SIZES[NUM_LAYERS];
    if (final_size == 1) {
        return (in_buf[0] >= 0.5f) ? 1 : 0;
    } else {
        int best = 0; float best_v = in_buf[0];
        for (int i = 1; i < final_size; i++)
            if (in_buf[i] > best_v) { best_v = in_buf[i]; best = i; }
        return best;
    }
}

/* normalization wrapper */
static inline void normalize_features(const float *in_raw, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float s = FEATURE_STD[i];
        out[i] = (in_raw[i] - FEATURE_MEAN[i]) / (s > 0.0f ? s : 1.0f);
    }
}

/* CSV logging (thread-safe) */
static inline void log_features_csv_locked(const struct sensor_key *key, const float f[NUM_FEATURES]) {
    if (!g_feat_csv) return;
    rte_spinlock_lock(&g_feat_csv_lock);
    /* write: unit_id,reg_addr,features... */
    fprintf(g_feat_csv, "%u,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            (unsigned)key->unit_id,
            (unsigned)key->reg_addr,
            (double)f[0], (double)f[1], (double)f[2], (double)f[3],
            (double)f[4], (double)f[5], (double)f[6], (double)f[7]);
    fflush(g_feat_csv);
    rte_spinlock_unlock(&g_feat_csv_lock);
}

/* --- sensor storage helpers --- */

static inline uint32_t allocate_sensor_index(struct worker_args *w) {
    if (w->next_free >= MAX_SENSORS_PER_CORE) return UINT32_MAX;
    return w->next_free++;
}

/* compute linear regression slope (units per second) */
static double compute_slope_seconds(const uint64_t *ts, const int32_t *y, int n, double hz) {
    double sumt = 0, sumy = 0, sumty = 0, sumt2 = 0;
    for (int i = 0; i < n; i++) {
        double t = (double)(ts[i] - ts[0]) / hz;
        double yi = (double)y[i];
        sumt += t;
        sumy += yi;
        sumty += t * yi;
        sumt2 += t * t;
    }
    double denom = (double)n * sumt2 - sumt * sumt;
    if (denom == 0.0) return 0.0;
    double slope = ((double)n * sumty - sumt * sumy) / denom;
    return slope;
}

static void compute_enhanced_features(const struct sensor_entry *e, float f[NUM_FEATURES], double hz) {
    /* 1. Original 8 features (FAST detection) */
    int32_t minv = INT32_MAX, maxv = INT32_MIN;
    double sum = 0.0;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        int32_t v = e->readings[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        sum += v;
    }
    double mean = sum / WINDOW_SIZE;

    uint64_t iat_min = UINT64_MAX, iat_max = 0;
    double iat_sum = 0.0;
    for (int i = 1; i < WINDOW_SIZE; i++) {
        uint64_t dt = (e->timestamps[i] > e->timestamps[i-1]) ? (e->timestamps[i] - e->timestamps[i-1]) : 0;
        if (dt < iat_min) iat_min = dt;
        if (dt > iat_max) iat_max = dt;
        iat_sum += (double)dt;
    }
    double mean_iat_us = (iat_sum / (WINDOW_SIZE - 1)) / hz * 1e6;
    double iat_min_us = (double)iat_min / hz * 1e6;
    double iat_max_us = (double)iat_max / hz * 1e6;

    double slope = compute_slope_seconds(e->timestamps, e->readings, WINDOW_SIZE, hz);

    /* Original 8 features */
    f[0] = (float)minv;
    f[1] = (float)maxv;
    f[2] = (float)mean;
    f[3] = (float)iat_min_us;
    f[4] = (float)iat_max_us;
    f[5] = (float)mean_iat_us;
    f[6] = (float)(maxv - minv);
    f[7] = (float)slope;
}

/* NEW: Compute CUSUM/EWMA enhanced features */
static void compute_cusum_ewma_features(struct sensor_entry *e, float f_extended[16]) {
    /* 1. Copy original 8 features */
    float original_features[8];
    compute_enhanced_features(e, original_features, g_tsc_hz);
    
    for (int i = 0; i < 8; i++) {
        f_extended[i] = original_features[i];
    }
    
    /* 2. EWMA Features (MEDIUM-term trends) */
    float current_ewma = 0.0f;
    if (e->sample_count > 0) {
        // Compute EWMA for current window
        current_ewma = (float)e->readings[0];
        for (int i = 1; i < WINDOW_SIZE; i++) {
            current_ewma = EWMA_ALPHA * (float)e->readings[i] + (1.0f - EWMA_ALPHA) * current_ewma;
        }
        
        // EWMA deviation from historical
        float ewma_deviation = current_ewma - e->ewma_value;
        
        f_extended[8] = current_ewma;                    // Current EWMA value
        f_extended[9] = ewma_deviation;                  // Absolute deviation
        f_extended[10] = fabsf(ewma_deviation) / (fabsf(e->ewma_value) + 1e-8f); // Relative change
        
        // Update persistent EWMA state
        e->ewma_value = current_ewma;
    } else {
        f_extended[8] = 0.0f;
        f_extended[9] = 0.0f;
        f_extended[10] = 0.0f;
    }
    
    /* 3. CUSUM Features (SLOW-term drift) */
    float cusum_pos = 0.0f, cusum_neg = 0.0f;
    
    if (e->sample_count >= WINDOW_SIZE) {
        // Update historical mean (running average)
        if (e->total_samples == 0) {
            e->historical_mean = f_extended[2]; // Use current mean
        } else {
            // Running average: update historical mean
            e->historical_mean = (e->historical_mean * e->total_samples + f_extended[2] * WINDOW_SIZE) 
                               / (e->total_samples + WINDOW_SIZE);
        }
        e->total_samples += WINDOW_SIZE;
        
        // Compute CUSUM for current window
        for (int i = 0; i < WINDOW_SIZE; i++) {
            float deviation = (float)e->readings[i] - e->historical_mean;
            cusum_pos = fmaxf(0.0f, cusum_pos + deviation - CUSUM_ALLOWANCE);
            cusum_neg = fminf(0.0f, cusum_neg + deviation + CUSUM_ALLOWANCE);
        }
        
        // Update persistent CUSUM state
        e->cusum_positive = cusum_pos;
        e->cusum_negative = cusum_neg;
        
        f_extended[11] = cusum_pos;                      // Positive CUSUM
        f_extended[12] = fabsf(cusum_neg);               // Negative CUSUM (absolute)
        f_extended[13] = fmaxf(cusum_pos, fabsf(cusum_neg)); // Maximum CUSUM
        f_extended[14] = (cusum_pos > CUSUM_THRESHOLD || fabsf(cusum_neg) > CUSUM_THRESHOLD) ? 1.0f : 0.0f; // Alarm
        
        // Trend consistency feature
        float trend_consistency = 0.0f;
        if (WINDOW_SIZE > 1) {
            int rising = 0, falling = 0;
            for (int i = 1; i < WINDOW_SIZE; i++) {
                if (e->readings[i] > e->readings[i-1]) rising++;
                else if (e->readings[i] < e->readings[i-1]) falling++;
            }
            trend_consistency = fabsf((float)(rising - falling) / (WINDOW_SIZE - 1));
        }
        f_extended[15] = trend_consistency;
    } else {
        f_extended[11] = 0.0f;
        f_extended[12] = 0.0f;
        f_extended[13] = 0.0f;
        f_extended[14] = 0.0f;
        f_extended[15] = 0.0f;
    }
}
/* compute features from a full sensor_entry */
static void compute_sensor_features(const struct sensor_entry *e, float f[NUM_FEATURES], double hz) {
    int32_t minv = INT32_MAX, maxv = INT32_MIN;
    double sum = 0.0;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        int32_t v = e->readings[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        sum += v;
    }
    double mean = sum / WINDOW_SIZE;

    uint64_t iat_min = UINT64_MAX, iat_max = 0;
    double iat_sum = 0.0;
    for (int i = 1; i < WINDOW_SIZE; i++) {
        uint64_t dt = (e->timestamps[i] > e->timestamps[i-1]) ? (e->timestamps[i] - e->timestamps[i-1]) : 0;
        if (dt < iat_min) iat_min = dt;
        if (dt > iat_max) iat_max = dt;
        iat_sum += (double)dt;
    }
    double mean_iat_us = (iat_sum / (WINDOW_SIZE - 1)) / hz * 1e6;
    double iat_min_us = (double)iat_min / hz * 1e6;
    double iat_max_us = (double)iat_max / hz * 1e6;

    double slope = compute_slope_seconds(e->timestamps, e->readings, WINDOW_SIZE, hz);

    f[0] = (float)minv;
    f[1] = (float)maxv;
    f[2] = (float)mean;
    f[3] = (float)iat_min_us;
    f[4] = (float)iat_max_us;
    f[5] = (float)mean_iat_us;
    f[6] = (float)(maxv - minv);
    f[7] = (float)slope;
}

/* Print sensor reading - called for every new reading */
static inline void print_sensor_reading(const struct sensor_key *key, int32_t reading, 
                                       uint64_t timestamp, struct worker_args *w, int is_32bit) {
    double timestamp_sec = (double)timestamp / g_tsc_hz;
    if (is_32bit) {
        //printf("Core %u: PLC unit_id=%u reg_addr=%u READING32=%d (0x%08X) time=%.6fs total_readings=%" PRIu64 "\n",
               //w->core_id, key->unit_id, key->reg_addr, reading, (uint32_t)reading,
               //timestamp_sec, w->total_readings);
    } else {
        //printf("Core %u: PLC unit_id=%u reg_addr=%u reading=%d (0x%04X) time=%.6fs total_readings=%" PRIu64 "\n",
               //w->core_id, key->unit_id, key->reg_addr, reading, (uint16_t)reading,
               //timestamp_sec, w->total_readings);
    }
}

/* handle incoming sensor reading - ENHANCED VERSION */
/* handle incoming sensor reading - ENHANCED VERSION */
static inline void handle_sensor_reading(const struct sensor_key *key,
                                         int32_t reading,
                                         uint64_t now_cycles,
                                         struct worker_args *w,
                                         int is_32bit)
{
    void *data_ptr = NULL;
    int ret = rte_hash_lookup_data(w->sensor_table, key, &data_ptr);
    uint32_t index;
    
    if (ret < 0) {
        index = allocate_sensor_index(w);
        if (index == UINT32_MAX) return;
        if (rte_hash_add_key_data(w->sensor_table, key, (void*)(uintptr_t)index) < 0) {
            w->next_free--;
            return;
        }
        /* initialize new entry with CUSUM/EWMA state */
        struct sensor_entry *e = &w->sensor_pool[index];
        memset(e, 0, sizeof(*e));
        e->sample_count = 0;
        e->write_index = 0;
        e->finalized = 0;
        e->ewma_value = (float)reading;  // Initialize EWMA with first reading
        e->historical_mean = (float)reading; // Initialize historical mean
        e->cusum_positive = 0.0f;
        e->cusum_negative = 0.0f;
        e->total_samples = 0;
        
        //printf("Core %u: NEW SENSOR STREAM - PLC unit_id=%u reg_addr=%u\n", 
               //w->core_id, key->unit_id, key->reg_addr);
    } else {
        index = (uint32_t)(uintptr_t)data_ptr;
    }

    struct sensor_entry *e = &w->sensor_pool[index];
    
    /* Print EVERY sensor reading */
    w->total_readings++;
    print_sensor_reading(key, reading, now_cycles, w, is_32bit);
    
    /* write into circular buffer */
    e->readings[e->write_index] = reading;
    e->timestamps[e->write_index] = now_cycles;
    e->sample_count++;
    e->write_index++;

    /* only run MLP when we have a full window */
    if (e->sample_count == WINDOW_SIZE) {
        /* compute ENHANCED features with CUSUM/EWMA */
                /* compute ENHANCED features with CUSUM/EWMA */
        float enhanced_features[16];  // 8 original + 8 enhanced

        uint64_t t0_feat = rte_rdtsc_precise();
        compute_cusum_ewma_features(e, enhanced_features);
        

        /* normalize */
        float features_scaled[16];
        normalize_features(enhanced_features, features_scaled, 16);
        uint64_t t1_feat = rte_rdtsc_precise();

        /* inference */
        uint64_t t0_inf = rte_rdtsc_precise();
        int pred = predict_mlp(features_scaled, w->buf_a, w->buf_b);
        uint64_t t1_inf = rte_rdtsc_precise();

        /* store timings if there's room */
        size_t idx = w->samples_count;
        if (idx < w->samples_capacity) {
            w->feat_cycles[idx] = t1_feat - t0_feat;     // feature extraction cycles
            w->infer_cycles[idx] = t1_inf - t0_inf;     // inference cycles
            w->sample_pred[idx] = pred;
            w->samples_count = idx + 1;
        }


        /* update counters & logging */
        if (pred == 0) w->pred0++; else w->pred1++;
        
        /* Enhanced logging - FIXED for 16 features */
        if (g_feat_csv) {
            rte_spinlock_lock(&g_feat_csv_lock);
            fprintf(g_feat_csv, "%u,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    (unsigned)key->unit_id, (unsigned)key->reg_addr,
                    (double)enhanced_features[0],  (double)enhanced_features[1],   // min, max
                    (double)enhanced_features[2],  (double)enhanced_features[3],   // mean, iat_min_us
                    (double)enhanced_features[4],  (double)enhanced_features[5],   // iat_max_us, mean_iat_us
                    (double)enhanced_features[6],  (double)enhanced_features[7],   // range, slope
                    (double)enhanced_features[8],  (double)enhanced_features[9],   // ewma_value, ewma_deviation
                    (double)enhanced_features[10], (double)enhanced_features[11],  // ewma_relative, cusum_positive
                    (double)enhanced_features[12], (double)enhanced_features[13],  // cusum_negative, cusum_max
                    (double)enhanced_features[14], (double)enhanced_features[15]   // cusum_alarm, trend_consistency
            );
            fflush(g_feat_csv);
            rte_spinlock_unlock(&g_feat_csv_lock);
        }

        /* print enhanced detection info */
        if (pred != 0) {
            //printf("Core %u: ANOMALY DETECTED - PLC unit_id=%u reg_addr=%u (pred=%d)\n",
                   //w->core_id, key->unit_id, key->reg_addr, pred);
            //printf("  Features: mean=%.3f range=%.3f ewma_dev=%.3f cusum_max=%.3f\n",
                   //enhanced_features[2], enhanced_features[6], 
                   //enhanced_features[9], enhanced_features[13]);
        }
        
        // RESET for next batch (samples 17-32, 33-48, etc.)
        e->sample_count = 0;
        e->write_index = 0;
        // Note: We DON'T clear CUSUM/EWMA state - they persist across windows!
    }
}

/* port_init implementation */
static inline int
port_init(uint16_t port, struct rte_mempool *mbuf_pool, uint16_t number_rings)
{
    struct rte_eth_dev_info dev_info;
    struct rte_eth_rxconf rxconf;
    struct rte_eth_txconf txconf;
    uint16_t nb_queue_pairs, rx_rings, tx_rings;
    int retval;
    uint16_t q;

    retval = rte_eth_dev_info_get(port, &dev_info);
    if (retval != 0) {
        printf("Error getting device info for port %u: %s\n", port, strerror(-retval));
        return retval;
    }

    nb_queue_pairs = number_rings;
    if (nb_queue_pairs > dev_info.max_rx_queues) nb_queue_pairs = dev_info.max_rx_queues;
    if (nb_queue_pairs > dev_info.max_tx_queues) nb_queue_pairs = dev_info.max_tx_queues;
    rx_rings = nb_queue_pairs;
    tx_rings = nb_queue_pairs;

    struct rte_eth_conf port_conf = {
        .rxmode = {
            .mq_mode  = RTE_ETH_MQ_RX_RSS,
            .offloads = RTE_ETH_RX_OFFLOAD_TIMESTAMP,
        },
        .rx_adv_conf = {
            .rss_conf = {
                .rss_key = NULL,
                .rss_hf  = RTE_ETH_RSS_IPV4 | RTE_ETH_RSS_TCP,
            },
        },
        .txmode = {
            .mq_mode = RTE_ETH_MQ_TX_NONE,
        },
    };

    if (!(dev_info.rx_offload_capa & RTE_ETH_RX_OFFLOAD_TIMESTAMP)) {
        port_conf.rxmode.offloads &= ~RTE_ETH_RX_OFFLOAD_TIMESTAMP;
    }

    port_conf.rx_adv_conf.rss_conf.rss_hf &= dev_info.flow_type_rss_offloads;
    if (port_conf.rx_adv_conf.rss_conf.rss_hf == 0) {
        port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_NONE;
    }

    retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
    if (retval < 0) return retval;

    uint16_t nb_rxd = RX_RING_SIZE;
    uint16_t nb_txd = TX_RING_SIZE;
    retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
    if (retval < 0) return retval;

    rxconf = dev_info.default_rxconf;
    for (q = 0; q < rx_rings; q++) {
        retval = rte_eth_rx_queue_setup(port, q, nb_rxd, rte_eth_dev_socket_id(port), &rxconf, mbuf_pool);
        if (retval < 0) return retval;
    }

    txconf = dev_info.default_txconf;
    txconf.offloads = port_conf.txmode.offloads;
    for (q = 0; q < tx_rings; q++) {
        retval = rte_eth_tx_queue_setup(port, q, nb_txd, rte_eth_dev_socket_id(port), &txconf);
        if (retval < 0) return retval;
    }

    retval = rte_eth_dev_start(port);
    if (retval < 0) return retval;
    rte_eth_promiscuous_enable(port);

    printf("Port %u initialized with %u queues\n", port, nb_queue_pairs);
    return 0;
}

/* proper lcore_main implementation */
static int lcore_worker(void *arg) {
    struct worker_args *w = (struct worker_args *)arg;
    struct rte_mempool *mbuf_pool = w->mbuf_pool;
    uint16_t port = w->port_id;
    uint16_t queue_id = w->queue_id;
    unsigned core_id = w->core_id;

    printf("Core %u running worker on port %u queue %u\n", core_id, port, queue_id);

    for (;;) {
        if (stop_requested) break;

        struct rte_mbuf *bufs[BURST_SIZE];
        uint16_t nb_rx = rte_eth_rx_burst(port, queue_id, bufs, BURST_SIZE);
        if (unlikely(nb_rx == 0)) {
            rte_pause();
            continue;
        }

        uint64_t t0_burst = rte_rdtsc_precise();
        rte_atomic64_add(&received_packets, nb_rx);

        for (uint16_t i = 0; i < nb_rx; i++) {
            struct rte_mbuf *m = bufs[i];
            uint32_t pkt_len_total = rte_pktmbuf_pkt_len(m);
            uint64_t packet_timestamp = rte_rdtsc_precise(); // CORRECTED: Get timestamp once per packet

            if (pkt_len_total < sizeof(struct rte_ether_hdr)) {
                rte_pktmbuf_free(m); bufs[i] = NULL; continue;
            }
            struct rte_ether_hdr *ethh = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);
            uint16_t ether_type = rte_be_to_cpu_16(ethh->ether_type);

            if (ether_type != RTE_ETHER_TYPE_IPV4) {
                continue;
            }

            uint32_t ip_offset = sizeof(struct rte_ether_hdr);
            if (pkt_len_total < ip_offset + sizeof(struct rte_ipv4_hdr)) {
                rte_pktmbuf_free(m); bufs[i] = NULL; continue;
            }
            struct rte_ipv4_hdr *ip4 = rte_pktmbuf_mtod_offset(m, struct rte_ipv4_hdr *, ip_offset);
            uint8_t ihl = ip4->version_ihl & RTE_IPV4_HDR_IHL_MASK;
            uint32_t ip_hdr_len = ihl * RTE_IPV4_IHL_MULTIPLIER;
            if (pkt_len_total < ip_offset + ip_hdr_len) { rte_pktmbuf_free(m); bufs[i] = NULL; continue; }

            uint8_t proto = ip4->next_proto_id;
            if (proto != IPPROTO_TCP) continue;

            uint32_t tcp_offset = ip_offset + ip_hdr_len;
            if (pkt_len_total < tcp_offset + sizeof(struct rte_tcp_hdr)) { rte_pktmbuf_free(m); bufs[i] = NULL; continue; }
            struct rte_tcp_hdr *tcp = rte_pktmbuf_mtod_offset(m, struct rte_tcp_hdr *, tcp_offset);

            uint16_t src_port = rte_be_to_cpu_16(tcp->src_port);
            uint16_t dst_port = rte_be_to_cpu_16(tcp->dst_port);
            uint8_t tcp_data_offset = (tcp->data_off >> 4) & 0x0F;
            uint32_t tcp_hdr_len = tcp_data_offset * 4;
            if (pkt_len_total < tcp_offset + tcp_hdr_len) { rte_pktmbuf_free(m); bufs[i] = NULL; continue; }

            /* Modbus-TCP detection (port 502 either source or dest) */
            if (dst_port != 502 && src_port != 502) {
                continue;
            }

            /* locate payload after TCP header */
            uint32_t payload_offset = tcp_offset + tcp_hdr_len;
            if (pkt_len_total <= payload_offset + sizeof(struct modbus_mbap)) {
                continue;
            }

            struct modbus_mbap *mbap = rte_pktmbuf_mtod_offset(m, struct modbus_mbap *, payload_offset);
            uint16_t trans_id = rte_be_to_cpu_16(mbap->trans_id);
            uint8_t unit_id = mbap->unit_id;
            
            uint32_t pdu_offset = payload_offset + sizeof(struct modbus_mbap);
            
            /* Check if this is a REQUEST (client → server) */
            if (src_port != 502 && dst_port == 502) {
                /* Modbus request */
                if (pkt_len_total >= pdu_offset + sizeof(struct modbus_request_pdu)) {
                    struct modbus_request_pdu *req_pdu = rte_pktmbuf_mtod_offset(m, struct modbus_request_pdu *, pdu_offset);
                    
                    if (req_pdu->function_code == 0x03 || req_pdu->function_code == 0x04) {
                        uint16_t start_addr = rte_be_to_cpu_16(req_pdu->start_addr);
                        uint16_t quantity = rte_be_to_cpu_16(req_pdu->quantity);
                        
                        //printf("Core %u: Modbus REQUEST trans_id=%u unit_id=%u start_addr=%u quantity=%u\n",
                               //w->core_id, trans_id, unit_id, start_addr, quantity);
                        
                        add_pending_request(trans_id, unit_id, start_addr, quantity);
                    }
                }
            }
            /* Check if this is a RESPONSE (server → client) */
            else if (src_port == 502 && dst_port != 502) {
                /* Modbus response */
                if (pkt_len_total >= pdu_offset + sizeof(struct modbus_response_pdu)) {
                    struct modbus_response_pdu *resp_pdu = rte_pktmbuf_mtod_offset(m, struct modbus_response_pdu *, pdu_offset);
                    
                    if (resp_pdu->function_code == 0x03 || resp_pdu->function_code == 0x04) {
                        struct pending_request req;
                        if (find_pending_request(trans_id, unit_id, &req)) {
                            /* Found matching request - process response */
                            uint8_t byte_count = resp_pdu->byte_count;
                            uint16_t num_registers = byte_count / 2;
                            
                            //printf("Core %u: Modbus RESPONSE trans_id=%u unit_id=%u start_addr=%u num_registers=%u byte_count=%u\n",
                                   //w->core_id, trans_id, unit_id, req.start_addr, num_registers, byte_count);
                            
                            /* Process all registers in the response */
                            for (uint16_t reg_idx = 0; reg_idx < num_registers; reg_idx++) {
                                uint16_t data_offset = pdu_offset + 2 + (reg_idx * 2); // +2 for function+byte_count
                                if (pkt_len_total < data_offset + 2) break;
                                
                                uint8_t *reg_data = rte_pktmbuf_mtod_offset(m, uint8_t *, data_offset);
                                uint16_t reg_hi = reg_data[0];
                                uint16_t reg_lo = reg_data[1];
                                uint16_t regval = (reg_hi << 8) | reg_lo;
                                
                                /* Check if this is part of a 32-bit value */
                                int is_32bit = 0;
                                int32_t sensor_value;
                                uint16_t actual_reg_addr = req.start_addr + reg_idx;
                                
                                /* Detect 32-bit values: if first register is 0 and we have multiple registers */
                                if (num_registers >= 2 && reg_idx < num_registers - 1) {
                                    uint16_t next_reg_hi = reg_data[2];
                                    uint16_t next_reg_lo = reg_data[3];
                                    uint16_t next_regval = (next_reg_hi << 8) | next_reg_lo;
                                    
                                    if (regval == 0 && next_regval != 0) {
                                        /* This is a 32-bit value */
                                        uint32_t combined_val = ((uint32_t)regval << 16) | next_regval;
                                        sensor_value = (int32_t)combined_val;
                                        is_32bit = 1;
                                        
                                        //printf("Core %u: 32-bit VALUE - reg_pair=[%u,%u] value=%d (0x%08X)\n",
                                              // w->core_id, actual_reg_addr, actual_reg_addr + 1, sensor_value, combined_val);
                                        
                                        reg_idx++; // Skip next register since we used it
                                    } else {
                                        sensor_value = (int16_t)regval; // Regular 16-bit
                                    }
                                } else {
                                    sensor_value = (int16_t)regval; // Regular 16-bit
                                }
                                
                                /* Create key with ACTUAL register address */
                                struct sensor_key key = { 
                                    .unit_id = unit_id,
                                    .reg_addr = actual_reg_addr
                                };
                                
                                handle_sensor_reading(&key, sensor_value, packet_timestamp, w, is_32bit);
                            }
                        } else {
                            //printf("Core %u: No matching request for trans_id=%u unit_id=%u\n",
                                   //w->core_id, trans_id, unit_id);
                        }
                    }
                }
            }
        }

        /* forward received mbufs that are non-freed */
        struct rte_mbuf *tx_bufs[BURST_SIZE];
        uint16_t tx_count = 0;
        for (uint16_t i = 0; i < nb_rx; i++) {
            if (bufs[i] != NULL) tx_bufs[tx_count++] = bufs[i];
        }
        if (tx_count > 0) {
            uint16_t nb_tx = rte_eth_tx_burst(port, queue_id, tx_bufs, tx_count);
            rte_atomic64_add(&processed_packets, nb_tx);
            if (nb_tx < tx_count) {
                for (uint16_t k = nb_tx; k < tx_count; k++) rte_pktmbuf_free(tx_bufs[k]);
            }
        }

         /* Record end of burst processing and store delta cycles */
        uint64_t t1_burst = rte_rdtsc_precise();
        size_t bidx = w->burst_count;
        if (bidx < w->burst_capacity) {
            w->burst_cycles[bidx] = t1_burst - t0_burst;
            w->burst_count = bidx + 1;
        }

    }

    return 0;
}

/* signal / cleanup */
static void sigint_handler(int signo) { stop_requested = 1; }

static void cleanup_and_exit(void) {
    if (g_feat_csv) { fflush(g_feat_csv); fclose(g_feat_csv); g_feat_csv = NULL; }

    uint64_t sum0 = 0, sum1 = 0;
    uint64_t total_readings = 0;
    for (unsigned core = 0; core < rte_lcore_count(); core++) {
        sum0 += worker_args[core].pred0;
        sum1 += worker_args[core].pred1;
        total_readings += worker_args[core].total_readings;
    }
    uint64_t total = sum0 + sum1;
    printf("\n=== Prediction summary ===\n");
    printf("class 0: %" PRIu64 "\n", sum0);
    printf("class 1: %" PRIu64 "\n", sum1);
    printf("total predictions: %" PRIu64 "\n", total);
    printf("total sensor readings: %" PRIu64 "\n", total_readings);

    uint16_t nb_ports = rte_eth_dev_count_avail();
    for (uint16_t pid = 0; pid < nb_ports; pid++) {
        int ret = rte_eth_dev_stop(pid);
        if (ret != 0) fprintf(stderr, "stop port %u err\n", pid);
        rte_eth_dev_close(pid);
    }

        /* write per-core timing CSVs */
    for (unsigned core = 0; core < rte_lcore_count(); core++) {
        struct worker_args *w = &worker_args[core];
        if (!w) continue;
        if (w->samples_count == 0) continue;

        char fname[64];
        snprintf(fname, sizeof(fname), "timings_core%u.csv", core);
        FILE *f = fopen(fname, "w");
        if (!f) {
            fprintf(stderr, "Cannot open %s for writing\n", fname);
            continue;
        }
        
        /* header: sample,feat_cycles,feat_ns,inf_cycles,inf_ns,pred */
        fprintf(f, "sample,feat_cycles,feat_ns,inf_cycles,inf_ns,pred\n");
        double tsc_hz = g_tsc_hz > 0.0 ? g_tsc_hz : (double)rte_get_tsc_hz();
        for (size_t i = 0; i < w->samples_count; i++) {
            uint64_t fc = w->feat_cycles[i];
            uint64_t ic = w->infer_cycles[i];
            double fns = ((double)fc / tsc_hz) * 1e9;
            double ins = ((double)ic / tsc_hz) * 1e9;
            fprintf(f, "%zu,%" PRIu64 ",%.2f,%" PRIu64 ",%.2f,%d\n",
                    i, fc, fns, ic, ins, w->sample_pred[i]);
        }

        /* write burst-level timing CSV */
        if (w->burst_count > 0) {
            char bfname[64];
            snprintf(bfname, sizeof(bfname), "burst_timings_core%u.csv", core);
            FILE *bf = fopen(bfname, "w");
            if (!bf) {
                fprintf(stderr, "Cannot open %s for writing\n", bfname);
            } else {
                fprintf(bf, "sample,burst_cycles,burst_ns\n");
                for (size_t j = 0; j < w->burst_count; j++) {
                    uint64_t bc = w->burst_cycles[j];
                    double bns = ((double)bc / tsc_hz) * 1e9;
                    fprintf(bf, "%zu,%" PRIu64 ",%.2f\n", j, bc, bns);
                }
                fclose(bf);
                printf("Wrote %s (%zu bursts)\n", bfname, w->burst_count);
            }
        }

        fclose(f);
        printf("Wrote %s (%zu samples)\n", fname, w->samples_count);
    }

   
    

    for (unsigned core = 0; core < rte_lcore_count(); core++) {
        struct worker_args *w = &worker_args[core];
        if (!w) continue;
        if (w->feat_cycles) { free(w->feat_cycles); w->feat_cycles = NULL; }
        if (w->infer_cycles) { free(w->infer_cycles); w->infer_cycles = NULL; }
        if (w->sample_pred) { free(w->sample_pred); w->sample_pred = NULL; }
        if (w->burst_cycles) { free(w->burst_cycles); w->burst_cycles = NULL; }
    }

    
    /* free per-core resources */
    unsigned total_lcores = rte_lcore_count();
    for (unsigned core = 0; core < total_lcores; core++) {
        struct worker_args *w = &worker_args[core];
        if (w->buf_a) free(w->buf_a); w->buf_a = NULL;
        if (w->buf_b) free(w->buf_b); w->buf_b = NULL;
        if (w->sensor_pool) rte_free(w->sensor_pool); w->sensor_pool = NULL;
        if (sensor_tables[core]) { rte_hash_free(sensor_tables[core]); sensor_tables[core] = NULL; }
    }


    /* free request tracker */
    if (g_request_tracker.requests) {
        rte_free(g_request_tracker.requests);
        g_request_tracker.requests = NULL;
    }

    rte_eal_cleanup();
}

/* main */
int main(int argc, char **argv) {
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) rte_panic("Cannot init EAL\n");
    argc -= ret; argv += ret;

    g_tsc_hz = (double)rte_get_tsc_hz();
    printf("TSC Hz: %.0f\n", g_tsc_hz);
    printf("DPDK version: %s\n", rte_version());

    /* Initialize request tracker */
    if (init_request_tracker(MAX_PENDING_REQUESTS) != 0) {
        rte_exit(EXIT_FAILURE, "Cannot initialize request tracker\n");
    }

    /* open CSV */
    /* Update the CSV header in main() function */
    /* open CSV */
    g_feat_csv = fopen("sensor_features.csv", "w");
    if (!g_feat_csv) rte_exit(EXIT_FAILURE, "Cannot open sensor_features.csv\n");
    setvbuf(g_feat_csv, NULL, _IOLBF, 0);
    // Update header for 16 features
    fprintf(g_feat_csv, "unit_id,reg_addr,min,max,mean,iat_min_us,iat_max_us,mean_iat_us,range,slope,ewma_value,ewma_deviation,ewma_relative,cusum_positive,cusum_negative,cusum_max,cusum_alarm,trend_consistency\n");

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    unsigned total_lcores = rte_lcore_count();

    rte_atomic64_init(&received_packets);
    rte_atomic64_init(&processed_packets);

    /* create per-core hash tables */
    struct rte_hash_parameters p = {
        .entries = MAX_SENSORS_PER_CORE,
        .key_len = sizeof(struct sensor_key),
        .hash_func = rte_jhash,
        .hash_func_init_val = 0,
        .socket_id = rte_socket_id(),
    };

    for (unsigned core = 0; core < total_lcores; core++) {
        char name[32];
        snprintf(name, sizeof(name), "stbl_%u", core);
        p.name = name;
        sensor_tables[core] = rte_hash_create(&p);
        if (!sensor_tables[core]) rte_exit(EXIT_FAILURE, "Cannot create sensor hash for core %u\n", core);
    }

    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0) rte_exit(EXIT_FAILURE, "No Ethernet ports\n");

    struct rte_mempool *mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL",
                                    NUM_MBUFS * nb_ports, MBUF_CACHE_SIZE, 0,
                                    RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (!mbuf_pool) rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

    /* init ports */
    uint16_t portid;
    RTE_ETH_FOREACH_DEV(portid) {
        if (port_init(portid, mbuf_pool, total_lcores) != 0)
            rte_exit(EXIT_FAILURE, "Cannot init port %" PRIu16 "\n", portid);
    }

    /* compute max neurons for buffer allocation */
    int max_neurons = 0;
    for (int i = 0; i <= NUM_LAYERS; i++) if (LAYER_SIZES[i] > max_neurons) max_neurons = LAYER_SIZES[i];

    /* initialize per-core worker args */
    uint16_t queue_id = 0;
    uint16_t base_port = 0;
    for (unsigned core_id = 0; core_id < total_lcores; core_id++) {
        struct worker_args *w = &worker_args[core_id];
        w->mbuf_pool = mbuf_pool;
        w->sensor_table = sensor_tables[core_id];

        /* allocate sensor pool */
        w->sensor_pool = rte_zmalloc_socket(NULL, sizeof(struct sensor_entry) * MAX_SENSORS_PER_CORE,
                                            RTE_CACHE_LINE_SIZE, rte_socket_id());
        if (!w->sensor_pool) rte_exit(EXIT_FAILURE, "Cannot allocate sensor pool\n");

        w->port_id = base_port;
        w->next_free = 0;
        w->queue_id = queue_id++;
        w->pred0 = w->pred1 = 0;
        w->total_readings = 0;
        w->core_id = core_id;

        /* allocate NEON buffers */
        if (posix_memalign((void**)&w->buf_a, 16, max_neurons * sizeof(float)) != 0 ||
            posix_memalign((void**)&w->buf_b, 16, max_neurons * sizeof(float)) != 0) {
            rte_exit(EXIT_FAILURE, "posix_memalign failed\n");
        }

        /* timing/instrumentation buffers */
        w->samples_capacity = MAX_TIMING_SAMPLES_PER_CORE;
        w->samples_count = 0;
        if (posix_memalign((void **)&w->feat_cycles, 64, w->samples_capacity * sizeof(uint64_t)) != 0 ||
            posix_memalign((void **)&w->infer_cycles, 64, w->samples_capacity * sizeof(uint64_t)) != 0 ||
            posix_memalign((void **)&w->sample_pred, 64, w->samples_capacity * sizeof(int32_t)) != 0) {
            rte_exit(EXIT_FAILURE, "posix_memalign failed for timing buffers\n");
        }
        memset(w->feat_cycles, 0, w->samples_capacity * sizeof(uint64_t));
        memset(w->infer_cycles, 0, w->samples_capacity * sizeof(uint64_t));
        memset(w->sample_pred,  0, w->samples_capacity * sizeof(int32_t));

        /* timing/instrumentation buffers */
        w->samples_capacity = MAX_TIMING_SAMPLES_PER_CORE;
        w->samples_count = 0;
        if (posix_memalign((void **)&w->feat_cycles, 64, w->samples_capacity * sizeof(uint64_t)) != 0 ||
            posix_memalign((void **)&w->infer_cycles, 64, w->samples_capacity * sizeof(uint64_t)) != 0 ||
            posix_memalign((void **)&w->sample_pred, 64, w->samples_capacity * sizeof(int32_t)) != 0) {
            rte_exit(EXIT_FAILURE, "posix_memalign failed for timing buffers\n");
        }
        memset(w->feat_cycles, 0, w->samples_capacity * sizeof(uint64_t));
        memset(w->infer_cycles, 0, w->samples_capacity * sizeof(uint64_t));
        memset(w->sample_pred,  0, w->samples_capacity * sizeof(int32_t));

        /* burst timing buffer (store one entry per RX burst) */
        w->burst_capacity = 262144; /* e.g., room for many bursts; tune as needed */
        w->burst_count = 0;
        if (posix_memalign((void **)&w->burst_cycles, 64, w->burst_capacity * sizeof(uint64_t)) != 0) {
            rte_exit(EXIT_FAILURE, "posix_memalign failed for burst timing buffer\n");
        }
        memset(w->burst_cycles, 0, w->burst_capacity * sizeof(uint64_t));


        /* launch remote workers */
        if (core_id != rte_get_main_lcore()) {
            rte_eal_remote_launch(lcore_worker, w, core_id);
        }
    }

    /* run main worker on master core */
    unsigned master = rte_get_main_lcore();
    struct worker_args *w_master = &worker_args[master];
    lcore_worker(w_master);

    /* wait remote lcores */
    rte_eal_mp_wait_lcore();

    /* cleanup */
    cleanup_and_exit();

    /* print counters */
    int64_t recv = rte_atomic64_read(&received_packets);
    int64_t proc = rte_atomic64_read(&processed_packets);
    printf("Received packets: %" PRId64 " Processed packets: %" PRId64 "\n", recv, proc);

    return 0;
}