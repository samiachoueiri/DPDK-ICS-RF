/* SPDX-License-Identifier: BSD-3-Clause
 * Safe RF loader + predictor (heap-allocated per-tree arrays)
 * Preserves original behavior but avoids stack overflow / OOB writes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <arm_neon.h>
#include <rte_eal.h>
#include <rte_cycles.h>
#include <jansson.h>

#include "test_data.h"   // must provide X_test[...][NUM_FEATURES] and TEST_N

//#define RF_MODEL_JSON  "rf_ws8_depth32_est1.json"
//#define RF_MODEL_JSON  "rf_ws8_depth32_est5.json"
//#define RF_MODEL_JSON  "rf_ws8_depth32_est10.json"
#define RF_MODEL_JSON  "rf_ws8_depth32_est25.json"
//#define RF_MODEL_JSON  "rf_ws8_depth32_est50.json"

#define NUM_FEATURES 8

/* Safety caps to avoid runaway allocations on DPU */
#define MAX_ALLOWED_ESTIMATORS        200     /* absolute cap on number of trees */
#define MAX_ALLOWED_NODES_PER_TREE  200000   /* cap on nodes per tree */
#define MAX_TOTAL_NODES            2000000   /* cap on total nodes across all trees */

/* Tree node struct (unchanged) */
typedef struct {
    int n_nodes;
    int left_child;
    int right_child;
    int feature;
    double threshold;
    int is_leaf;
    int class_label;
} TreeNode;

/* RandomForest now uses heap allocations for tree arrays */
typedef struct {
    int n_estimators;
    int max_depth;
    double feature_importances[NUM_FEATURES];
    TreeNode **trees;        /* length n_estimators */
    int *tree_node_counts;   /* length n_estimators */
} RandomForest;

/* Safe helpers */
static void dump_bytes(const void *ptr, size_t n) {
    const unsigned char *b = (const unsigned char *)ptr;
    for (size_t i = 0; i < n; ++i) {
        if ((i % 16) == 0) printf("%08zx: ", i);
        printf("%02x ", b[i]);
        if ((i % 16) == 15) printf("\n");
    }
    if (n % 16) printf("\n");
}

static void free_rf(RandomForest *rf) {
    if (!rf) return;
    if (rf->trees) {
        for (int i = 0; i < rf->n_estimators; ++i) {
            free(rf->trees[i]);
            rf->trees[i] = NULL;
        }
        free(rf->trees);
        rf->trees = NULL;
    }
    if (rf->tree_node_counts) {
        free(rf->tree_node_counts);
        rf->tree_node_counts = NULL;
    }
}

/* Load RF from JSON safely (allocates per-tree arrays on heap) */
int load_rf_model(const char *filename, RandomForest *rf) {
    if (!filename || !rf) return -1;

    json_error_t error;
    json_t *root = json_load_file(filename, 0, &error);
    if (!root) {
        fprintf(stderr, "Error loading %s: %s\n", filename, error.text);
        return -1;
    }

    /* read n_estimators */
    json_t *je = json_object_get(root, "n_estimators");
    if (!je || !json_is_integer(je)) {
        fprintf(stderr, "Model JSON missing valid n_estimators\n");
        json_decref(root);
        return -1;
    }
    int n_estimators = (int)json_integer_value(je);
    if (n_estimators <= 0 || n_estimators > MAX_ALLOWED_ESTIMATORS) {
        fprintf(stderr, "n_estimators %d out of allowed 1..%d\n",
                n_estimators, MAX_ALLOWED_ESTIMATORS);
        json_decref(root);
        return -1;
    }
    rf->n_estimators = n_estimators;

    /* max depth */
    je = json_object_get(root, "max_depth");
    if (je && json_is_integer(je))
        rf->max_depth = (int)json_integer_value(je);
    else
        rf->max_depth = 0;

    /* feature importances */
    json_t *fi = json_object_get(root, "feature_importances");
    if (!fi || !json_is_array(fi)) {
        fprintf(stderr, "Model JSON missing feature_importances array\n");
        json_decref(root);
        return -1;
    }
    for (int i = 0; i < NUM_FEATURES; ++i) {
        json_t *v = json_array_get(fi, i);
        rf->feature_importances[i] = v ? json_real_value(v) : 0.0;
    }

    /* Get estimators array */
    json_t *estimators = json_object_get(root, "estimators");
    if (!estimators || !json_is_array(estimators)) {
        fprintf(stderr, "Model JSON missing estimators array\n");
        json_decref(root);
        return -1;
    }

    /* Pre-check total nodes to estimate memory and protect from huge allocations */
    size_t total_nodes = 0;
    size_t estimators_size = (size_t)json_array_size(estimators);
    for (size_t idx = 0; idx < estimators_size && idx < (size_t)rf->n_estimators; ++idx) {
        json_t *tn = json_array_get(estimators, idx);
        if (!tn) continue;
        json_t *jn = json_object_get(tn, "n_nodes");
        int tn_nodes = jn ? (int)json_integer_value(jn) : 0;
        if (tn_nodes < 0) tn_nodes = 0;
        total_nodes += (size_t)tn_nodes;
        if (tn_nodes > MAX_ALLOWED_NODES_PER_TREE) {
            fprintf(stderr, "Tree %zu n_nodes %d > allowed %d\n",
                    idx, tn_nodes, MAX_ALLOWED_NODES_PER_TREE);
            json_decref(root);
            return -1;
        }
        if (total_nodes > MAX_TOTAL_NODES) {
            fprintf(stderr, "Total nodes %zu exceed allowed %d\n",
                    total_nodes, MAX_TOTAL_NODES);
            json_decref(root);
            return -1;
        }
    }

    size_t bytes_needed = total_nodes * sizeof(TreeNode);
    double mib = (double)bytes_needed / (1024.0*1024.0);
    printf("Model: estimators=%d total_nodes=%zu approx %.2f MiB\n",
           rf->n_estimators, total_nodes, mib);

    /* allocate arrays of pointers and counts */
    rf->trees = calloc((size_t)rf->n_estimators, sizeof(TreeNode *));
    if (!rf->trees) { perror("calloc trees"); json_decref(root); return -1; }
    rf->tree_node_counts = calloc((size_t)rf->n_estimators, sizeof(int));
    if (!rf->tree_node_counts) { perror("calloc counts"); free(rf->trees); rf->trees = NULL; json_decref(root); return -1; }

    /* Now parse each estimator and allocate per-tree node arrays */
    size_t idx;
    json_t *tn;
    json_array_foreach(estimators, idx, tn) {
        if ((int)idx >= rf->n_estimators) break;
        json_t *jn_nodes = json_object_get(tn, "n_nodes");
        int n_nodes = jn_nodes ? (int)json_integer_value(jn_nodes) : 0;
        if (n_nodes <= 0) {
            /* empty tree? allocate zero length (NULL) and mark zero */
            rf->trees[idx] = NULL;
            rf->tree_node_counts[idx] = 0;
            continue;
        }
        if (n_nodes > MAX_ALLOWED_NODES_PER_TREE) {
            fprintf(stderr, "Refusing to allocate tree %zu with n_nodes %d\n", idx, n_nodes);
            /* cleanup */
            for (int j = 0; j < (int)idx; ++j) free(rf->trees[j]);
            free(rf->trees); rf->trees = NULL;
            free(rf->tree_node_counts); rf->tree_node_counts = NULL;
            json_decref(root);
            return -1;
        }

        TreeNode *tree = malloc((size_t)n_nodes * sizeof(TreeNode));
        if (!tree) {
            perror("malloc tree");
            for (int j = 0; j < (int)idx; ++j) free(rf->trees[j]);
            free(rf->trees); rf->trees = NULL;
            free(rf->tree_node_counts); rf->tree_node_counts = NULL;
            json_decref(root);
            return -1;
        }
        /* record */
        rf->trees[idx] = tree;
        rf->tree_node_counts[idx] = n_nodes;

        /* arrays for this tree */
        json_t *left  = json_object_get(tn, "children_left");
        json_t *right = json_object_get(tn, "children_right");
        json_t *feat  = json_object_get(tn, "feature");
        json_t *th    = json_object_get(tn, "threshold");
        json_t *cl    = json_object_get(tn, "class_label");
        json_t *leaf  = json_object_get(tn, "leaves");

        /* fill nodes */
        for (int i = 0; i < n_nodes; ++i) {
            tree[i].n_nodes = n_nodes;
            json_t *v;
            v = left  ? json_array_get(left, i) : NULL;
            tree[i].left_child  = v ? (int)json_integer_value(v) : -1;
            v = right ? json_array_get(right, i) : NULL;
            tree[i].right_child = v ? (int)json_integer_value(v) : -1;
            v = feat  ? json_array_get(feat, i) : NULL;
            tree[i].feature = v ? (int)json_integer_value(v) : -1;
            v = th ? json_array_get(th, i) : NULL;
            tree[i].threshold = v ? json_real_value(v) : 0.0;
            v = cl ? json_array_get(cl, i) : NULL;
            tree[i].class_label = v ? (int)json_integer_value(v) : -1;
            v = leaf ? json_array_get(leaf, i) : NULL;
            tree[i].is_leaf = v ? (int)json_integer_value(v) : 0;
        }
    }

    json_decref(root);
    return 0;
}

/* safe recursive predictor with bounds guard.
 * n_nodes is used to ensure idx is within tree size and to avoid infinite recursion.
 */
static int predict_tree(const TreeNode *tree, int n_nodes, const float *sample, int idx) {
    if (!tree) return -1;
    if (idx < 0 || idx >= n_nodes) return -1; /* invalid index */
    if (tree[idx].is_leaf) return tree[idx].class_label;
    int feat = tree[idx].feature;
    double thr = tree[idx].threshold;
    int next = (sample && feat >= 0 && feat < NUM_FEATURES && sample[feat] <= (float)thr)
               ? tree[idx].left_child
               : tree[idx].right_child;
    if (next < 0 || next >= n_nodes) {
        /* defensive: if child index is invalid, try to treat current node as leaf */
        return tree[idx].class_label;
    }
    return predict_tree(tree, n_nodes, sample, next);
}

/* majority-vote RF predictor */
int predict_rf(const RandomForest *rf, const float *sample) {
    if (!rf) return -1;
    int counts[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; ++i) counts[i] = 0;
    for (int e = 0; e < rf->n_estimators; ++e) {
        TreeNode *tree = rf->trees[e];
        int n_nodes = rf->tree_node_counts[e];
        if (!tree || n_nodes <= 0) continue;
        int p = predict_tree(tree, n_nodes, sample, 0);
        if (p >= 0 && p < NUM_FEATURES) counts[p]++;
    }
    /* majority vote / argmax */
    int best = 0, maxc = counts[0];
    for (int i = 1; i < NUM_FEATURES; ++i) {
        if (counts[i] > maxc) { maxc = counts[i]; best = i; }
    }
    return best;
}

/* main */
int main(int argc, char **argv) {
    if (rte_eal_init(argc, argv) < 0)
        rte_exit(EXIT_FAILURE, "EAL init failed\n");

    RandomForest *rf = calloc(1, sizeof(*rf));
    if (!rf) rte_exit(EXIT_FAILURE, "calloc rf failed\n");

    if (load_rf_model(RF_MODEL_JSON, rf) != 0) {
        fprintf(stderr, "Failed to load RF model\n");
        free_rf(rf);
        free(rf);
        return EXIT_FAILURE;
    }

    size_t rf_size_est = 0;
    for (int i = 0; i < rf->n_estimators; ++i)
        rf_size_est += (size_t)rf->tree_node_counts[i] * sizeof(TreeNode);
    printf("Loaded RF: %d estimators, approx %.2f MiB for nodes\n",
           rf->n_estimators, (double)rf_size_est / (1024.0*1024.0));

    /* optional: quick byte-dump head */
    const size_t HEAD_BYTES = 256;
    printf("Dumping first %zu bytes of RF struct (heap metadata not included):\n", HEAD_BYTES);
    /* we will dump the structure header (RandomForest) memory */
    dump_bytes(rf, HEAD_BYTES);

    /* Benchmark loop using test_data.h's X_test and TEST_N */
    FILE *out = fopen("latencies.csv","w");
    if (!out) {
        perror("fopen latencies.csv");
        free_rf(rf);
        free(rf);
        return EXIT_FAILURE;
    }
    fprintf(out, "iter,rf_ns\n");

    srand((unsigned)rte_get_tsc_cycles());
    const uint64_t hz = rte_get_tsc_hz();
    
    uint64_t correct = 0;
    uint64_t total   = 0;

    for (int it = 0; it < TEST_N; ++it) {
        uint64_t t0 = rte_rdtsc_precise();
        int rf_pred = predict_rf(rf, X_test[it]); (void)rf_pred;
        uint64_t t1 = rte_rdtsc_precise();
        double rf_ns = (double)(t1 - t0) * 1e9 / (double)hz;

        /* accuracy check */
        int y_true = atoi(y_expected_str[it]);  // convert "0"/"1"/... to int
        if (rf_pred == y_true) correct++;
        
        total++;

        fprintf(out, "%d,%.2f\n", it, rf_ns);
    }

    double accuracy = (total > 0) ? ((double)correct / (double)total) : 0.0;
    printf("RF online accuracy: %.6f (%" PRIu64 "/%" PRIu64 ")\n", accuracy, correct, total);

    fclose(out);

    /* cleanup */
    free_rf(rf);
    free(rf);

    return 0;
}
