/* SPDX-License-Identifier: BSD-3-Clause
 * Combined RF and NEON-accelerated MLP inference latency benchmark.
 * Measures per-iteration latency of Random Forest (from JSON) vs.
 * NEON-optimized MLP (from header) over ITERATIONS random samples.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>
#include <arm_neon.h>
#include <rte_eal.h>
#include <rte_cycles.h>
#include <jansson.h>            // JSON parsing

#include "test_data.h"


//#define RF_MODEL_JSON  "rf_ws8_depth32_est1.json"
//#define RF_MODEL_JSON  "rf_ws8_depth32_est5.json"
#define RF_MODEL_JSON  "rf_ws8_depth32_est10.json"
// #define RF_MODEL_JSON  "rf_ws8_depth32_est25.json"
// #define RF_MODEL_JSON  "rf_ws8_depth32_est50.json"


#define MIN_PKT_SIZE  64.0f
#define MAX_PKT_SIZE  9000.0f
#define MAX_LEN_RANGE (MAX_PKT_SIZE - MIN_PKT_SIZE)

// Random Forest limits
#define MAX_TREES 10
#define MAX_NODES 50000 // 25000 30000

#define NUM_FEATURES 8 

// Random Forest structures & JSON loader

typedef struct {
    int n_nodes;
    int left_child;
    int right_child;
    int feature;
    double threshold;
    int is_leaf;
    int class_label;
} TreeNode;

typedef struct {
    int n_estimators;
    int max_depth;
    double feature_importances[NUM_FEATURES];
    TreeNode trees[MAX_TREES][MAX_NODES];
} RandomForest;

// Load RF from JSON file
int load_rf_model(const char *filename, RandomForest *rf) {
    json_error_t error;
    json_t *root = json_load_file(filename, 0, &error);
    if (!root) {
        fprintf(stderr, "Error loading %s: %s\n", filename, error.text);
        return -1;
    }
    json_t *je = json_object_get(root, "n_estimators");
    rf->n_estimators = json_integer_value(je);
    je = json_object_get(root, "max_depth");
    rf->max_depth = json_integer_value(je);

    // feature importances
    je = json_object_get(root, "feature_importances");
    for (int i = 0; i < NUM_FEATURES; i++) {
        rf->feature_importances[i] =
            json_real_value(json_array_get(je, i));
    }

    // parse trees
    json_t *estimators = json_object_get(root, "estimators");
    size_t idx;
    json_t *tn;
    json_array_foreach(estimators, idx, tn) {
        TreeNode *tree = rf->trees[idx];
        int n_nodes = json_integer_value(
            json_object_get(tn, "n_nodes"));
        // arrays
        json_t *left  = json_object_get(tn, "children_left");
        json_t *right = json_object_get(tn, "children_right");
        json_t *feat  = json_object_get(tn, "feature");
        json_t *th    = json_object_get(tn, "threshold");
        json_t *cl    = json_object_get(tn, "class_label");
        json_t *leaf  = json_object_get(tn, "leaves");
        for (int i = 0; i < n_nodes; i++) {
            tree[i].n_nodes     = n_nodes;
            tree[i].left_child  = json_integer_value(
                                   json_array_get(left, i));
            tree[i].right_child = json_integer_value(
                                   json_array_get(right, i));
            tree[i].feature     = json_integer_value(
                                   json_array_get(feat, i));
            tree[i].threshold   = json_real_value(
                                   json_array_get(th, i));
            tree[i].class_label = json_integer_value(
                                   json_array_get(cl, i));
            tree[i].is_leaf     = json_integer_value(
                                   json_array_get(leaf, i));
        }
    }

    json_decref(root);
    return 0;
}

// recursive tree predictor
static int predict_tree(const TreeNode *tree, const float *sample, int idx) {
    if (tree[idx].is_leaf)
        return tree[idx].class_label;
    if (sample[tree[idx].feature] <= tree[idx].threshold)
        return predict_tree(tree, sample, tree[idx].left_child);
    else
        return predict_tree(tree, sample, tree[idx].right_child);
}

// majority‐vote RF predictor
int predict_rf(const RandomForest *rf, const float *sample) {
    int counts[NUM_FEATURES] = {0};
    for (int e = 0; e < rf->n_estimators; e++) {
        int p = predict_tree(rf->trees[e], sample, 0);
        if (p >= 0 && p < NUM_FEATURES) counts[p]++;
    }
    // find max
    int best = 0, maxc = counts[0];
    for (int i = 1; i < NUM_FEATURES; i++) {
        if (counts[i] > maxc) {
            maxc = counts[i];
            best = i;
        }
    }
    return best;
}

static void dump_bytes(const void *ptr, size_t n) {
    const unsigned char *b = (const unsigned char *)ptr;
    for (size_t i = 0; i < n; ++i) {
        if ((i % 16) == 0) printf("%08zx: ", i);
        printf("%02x ", b[i]);
        if ((i % 16) == 15) printf("\n");
    }
    if (n % 16) printf("\n");
}

int main(int argc, char **argv) {

    if (rte_eal_init(argc, argv) < 0)
        rte_exit(EXIT_FAILURE, "EAL init failed\n");

   
    // 1) Load RF model
    RandomForest rf;
    if (load_rf_model(RF_MODEL_JSON, &rf) != 0)
        return -1;

    size_t rf_size = sizeof(RandomForest);
    printf("RandomForest @ %p size=%zu bytes\n", (void*)&rf, rf_size);


    // 2) Open CSV
    FILE *out = fopen("latencies.csv","w");
    if (!out) rte_exit(EXIT_FAILURE, "Cannot open latencies.csv\n");
    fprintf(out, "iter,rf_ns\n");

    srand((unsigned)rte_get_tsc_cycles());
    const uint64_t hz = rte_get_tsc_hz();

    // 3) Benchmark loop
    for (int it = 0; it < TEST_N; it++) {
        // a) time RF
        uint64_t t0 = rte_rdtsc_precise();
        int rf_pred = predict_rf(&rf, X_test[it]);
        uint64_t t1 = rte_rdtsc_precise();
        double rf_ns = (double)(t1 - t0)*1e9/hz;

        // b) write CSV
        fprintf(out, "%d,%.2f\n", it, rf_ns);
    }

    fclose(out);
    return 0;
}