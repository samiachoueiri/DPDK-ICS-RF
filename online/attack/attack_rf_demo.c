/* SPDX-License-Identifier: BSD-3-Clause
 * Safe RF loader + predictor (heap-allocated per-tree arrays)
 * Adds pretty demo printing for classification tracing.
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
#include <unistd.h>

#ifdef __GLIBC__
#include <malloc.h>
#endif

#include "test_data.h"

#define RF_MODEL_JSON  "random_forest_ws8_trees1_depth8.json"
//#define RF_MODEL_JSON  "random_forest_ws8_trees5_depth8.json"
//#define RF_MODEL_JSON  "random_forest_ws8_trees10_depth8.json"
//#define RF_MODEL_JSON  "random_forest_ws8_trees25_depth8.json"
//#define RF_MODEL_JSON  "random_forest_ws8_trees50_depth8.json"

#define NUM_FEATURES 8
#define DEMO_VERBOSE_SAMPLES 3

/* Safety caps to avoid runaway allocations on DPU */
#define MAX_ALLOWED_ESTIMATORS        200
#define MAX_ALLOWED_NODES_PER_TREE  200000
#define MAX_TOTAL_NODES            2000000

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
    TreeNode **trees;        /* length n_estimators */
    int *tree_node_counts;   /* length n_estimators */
} RandomForest;

/* -------------------- helpers -------------------- */

static void dump_bytes(const void *ptr, size_t n) {
    const unsigned char *b = (const unsigned char *)ptr;
    for (size_t i = 0; i < n; ++i) {
        if ((i % 16) == 0) printf("%08zx: ", i);
        printf("%02x ", b[i]);
        if ((i % 16) == 15) printf("\n");
    }
    if (n % 16) printf("\n");
}

static void print_bytes_human(size_t bytes, const char *label) {
    double kib = (double)bytes / 1024.0;
    double mib = kib / 1024.0;
    printf("%s: %zu bytes (%.2f KiB, %.4f MiB)\n", label, bytes, kib, mib);
}

static void print_divider(void) {
    printf("\n============================================================\n");
}

static void print_sample_features(const float *sample) {
    printf("Features: ");
    for (int i = 0; i < NUM_FEATURES; ++i) {
        printf("x[%d]=%.3f%s", i, sample[i], (i == NUM_FEATURES - 1) ? "\n" : " | ");
    }
}

static void print_vote_tally(const int *counts) {
    printf("Vote: ");
    for (int i = 0; i < 2; ++i) {
        printf("%d:%d%s", i, counts[i], (i == NUM_FEATURES - 1) ? "\n" : "  ");
    }
}

static void print_indent(int depth) {
    for (int i = 0; i < depth; ++i) printf("  ");
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

/* -------------------- predictors -------------------- */

/* Original non-verbose tree predictor */
static int predict_tree(const TreeNode *tree, int n_nodes, const float *sample, int idx) {
    if (!tree) return -1;
    if (idx < 0 || idx >= n_nodes) return -1;

    if (tree[idx].is_leaf) return tree[idx].class_label;

    int feat = tree[idx].feature;
    double thr = tree[idx].threshold;

    int next = (sample && feat >= 0 && feat < NUM_FEATURES && sample[feat] <= (float)thr)
               ? tree[idx].left_child
               : tree[idx].right_child;

    if (next < 0 || next >= n_nodes) {
        return tree[idx].class_label;
    }

    return predict_tree(tree, n_nodes, sample, next);
}

static int predict_tree_verbose(const TreeNode *tree, int n_nodes, const float *sample, int idx, int depth) {
    if (!tree) {
        print_indent(depth);
        printf("Invalid tree pointer\n");
        return -1;
    }
    if (idx < 0 || idx >= n_nodes) {
        print_indent(depth);
        printf("Invalid node index %d (tree size=%d)\n", idx, n_nodes);
        return -1;
    }

    const TreeNode *node = &tree[idx];

    print_indent(depth);
    printf("Node %d: ", idx);

    if (node->is_leaf) {
        printf("LEAF -> class %d\n", node->class_label);
        return node->class_label;
    }

    int feat = node->feature;
    double thr = node->threshold;
    float val = (sample && feat >= 0 && feat < NUM_FEATURES) ? sample[feat] : 0.0f;
    int go_left = (sample && feat >= 0 && feat < NUM_FEATURES && val <= (float)thr);

    printf("x[%d]=%.3f %s %.3f -> %s\n",
           feat, val, go_left ? "<=" : ">", thr, go_left ? "left" : "right");

    int next = go_left ? node->left_child : node->right_child;

    if (next < 0 || next >= n_nodes) {
        print_indent(depth + 1);
        printf("Invalid child index %d, treating current node as leaf -> class %d\n",
               next, node->class_label);
        return node->class_label;
    }

    return predict_tree_verbose(tree, n_nodes, sample, next, depth + 1);
}

static int predict_rf_demo(const RandomForest *rf, const float *sample, int verbose) {
    if (!rf) return -1;

    int counts[2] = {0, 0};  // class 0 and class 1 only

    for (int e = 0; e < rf->n_estimators; ++e) {
        TreeNode *tree = rf->trees[e];
        int n_nodes = rf->tree_node_counts[e];
        if (!tree || n_nodes <= 0) continue;

        if (verbose) {
            printf("\nTree %d/%d:\n", e + 1, rf->n_estimators);
        }

        int p = verbose
              ? predict_tree_verbose(tree, n_nodes, sample, 0, 1)
              : predict_tree(tree, n_nodes, sample, 0);

        if (verbose) {
            printf("Tree %d prediction: %d\n", e + 1, p);
        }

        if (p == 0 || p == 1) {
            counts[p]++;
        }

        if (verbose) {
            print_vote_tally(counts);
        }
    }

    int best = (counts[1] > counts[0]) ? 1 : 0;
    int maxc = counts[best];

    if (verbose) {
        printf("\nFinal majority vote: class %d (%d votes)\n", best, maxc);
        print_vote_tally(counts);
    }

    return best;
}

/* -------------------- loader -------------------- */

int load_rf_model(const char *filename, RandomForest *rf) {
    if (!filename || !rf) return -1;

    json_error_t error;
    json_t *root = json_load_file(filename, 0, &error);
    if (!root) {
        fprintf(stderr, "Error loading %s: %s\n", filename, error.text);
        return -1;
    }

    /* n_estimators */
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

    /* estimators */
    json_t *estimators = json_object_get(root, "estimators");
    if (!estimators || !json_is_array(estimators)) {
        fprintf(stderr, "Model JSON missing estimators array\n");
        json_decref(root);
        return -1;
    }

    /* pre-check total nodes */
    size_t total_nodes = 0;
    size_t estimators_size = (size_t)json_array_size(estimators);
    for (size_t idx = 0; idx < estimators_size && idx < (size_t)rf->n_estimators; ++idx) {
        json_t *tn = json_array_get(estimators, idx);
        if (!tn) continue;

        json_t *jn = json_object_get(tn, "n_nodes");
        int tn_nodes = jn ? (int)json_integer_value(jn) : 0;
        if (tn_nodes < 0) tn_nodes = 0;

        if (tn_nodes > MAX_ALLOWED_NODES_PER_TREE) {
            fprintf(stderr, "Tree %zu n_nodes %d > allowed %d\n",
                    idx, tn_nodes, MAX_ALLOWED_NODES_PER_TREE);
            json_decref(root);
            return -1;
        }

        total_nodes += (size_t)tn_nodes;
        if (total_nodes > MAX_TOTAL_NODES) {
            fprintf(stderr, "Total nodes %zu exceed allowed %d\n",
                    total_nodes, MAX_TOTAL_NODES);
            json_decref(root);
            return -1;
        }
    }

    size_t bytes_needed = total_nodes * sizeof(TreeNode);
    print_bytes_human(bytes_needed, "Estimated node storage");

    /* allocate arrays */
    rf->trees = calloc((size_t)rf->n_estimators, sizeof(TreeNode *));
    if (!rf->trees) {
        perror("calloc trees");
        json_decref(root);
        return -1;
    }

    rf->tree_node_counts = calloc((size_t)rf->n_estimators, sizeof(int));
    if (!rf->tree_node_counts) {
        perror("calloc counts");
        free(rf->trees);
        rf->trees = NULL;
        json_decref(root);
        return -1;
    }

    /* parse each estimator */
    size_t idx;
    json_t *tn;
    json_array_foreach(estimators, idx, tn) {
        if ((int)idx >= rf->n_estimators) break;

        json_t *jn_nodes = json_object_get(tn, "n_nodes");
        int n_nodes = jn_nodes ? (int)json_integer_value(jn_nodes) : 0;

        if (n_nodes <= 0) {
            rf->trees[idx] = NULL;
            rf->tree_node_counts[idx] = 0;
            printf("Tree %zu: empty\n", idx);
            continue;
        }

        if (n_nodes > MAX_ALLOWED_NODES_PER_TREE) {
            fprintf(stderr, "Refusing to allocate tree %zu with n_nodes %d\n", idx, n_nodes);
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

        rf->trees[idx] = tree;
        rf->tree_node_counts[idx] = n_nodes;

        size_t tree_bytes = (size_t)n_nodes * sizeof(TreeNode);
        printf("Tree %zu: %d nodes, %zu bytes (%.2f KiB)\n",
               idx, n_nodes, tree_bytes, (double)tree_bytes / 1024.0);

#ifdef __GLIBC__
        printf("Tree %zu usable allocation: %zu bytes\n",
               idx, (size_t)malloc_usable_size(tree));
#endif

        json_t *left  = json_object_get(tn, "children_left");
        json_t *right = json_object_get(tn, "children_right");
        json_t *feat  = json_object_get(tn, "feature");
        json_t *th    = json_object_get(tn, "threshold");
        json_t *cl    = json_object_get(tn, "class_label");
        json_t *leaf  = json_object_get(tn, "leaves");

        for (int i = 0; i < n_nodes; ++i) {
            tree[i].n_nodes = n_nodes;

            json_t *v;
            v = left ? json_array_get(left, i) : NULL;
            tree[i].left_child = v ? (int)json_integer_value(v) : -1;

            v = right ? json_array_get(right, i) : NULL;
            tree[i].right_child = v ? (int)json_integer_value(v) : -1;

            v = feat ? json_array_get(feat, i) : NULL;
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

/* -------------------- main -------------------- */

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

    size_t rf_node_bytes = 0;
    for (int i = 0; i < rf->n_estimators; ++i) {
        rf_node_bytes += (size_t)rf->tree_node_counts[i] * sizeof(TreeNode);
    }

    size_t rf_ptr_bytes = (size_t)rf->n_estimators * sizeof(TreeNode *);
    size_t rf_cnt_bytes = (size_t)rf->n_estimators * sizeof(int);
    size_t rf_total_bytes = rf_node_bytes + rf_ptr_bytes + rf_cnt_bytes;

    print_bytes_human(rf_node_bytes, "Tree node storage");
    print_bytes_human(rf_ptr_bytes,  "Tree pointer array");
    print_bytes_human(rf_cnt_bytes,   "Tree count array");
    print_bytes_human(rf_total_bytes, "Estimated RF heap footprint");

    const size_t HEAD_BYTES = 256;
    printf("Dumping first %zu bytes of RF struct (heap metadata not included):\n", HEAD_BYTES);
    dump_bytes(rf, HEAD_BYTES);

    FILE *out = fopen("latencies.csv", "w");
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
    uint64_t total = 0;

    for (int it = 0; it < 20 && it < TEST_N; ++it) {
        int y_true = atoi(y_expected_str[it]);
        int verbose = 1;   /* show demo output for every sample */

        print_divider();
        printf("Sample %d / %zu\n", it + 1, TEST_N);
        print_sample_features(X_test[it]);
        printf("True label: %d\n", y_true);

        uint64_t t0 = rte_rdtsc_precise();
        int rf_pred = predict_rf_demo(rf, X_test[it], verbose);
        uint64_t t1 = rte_rdtsc_precise();

        double rf_ns = (double)(t1 - t0) * 1e9 / (double)hz;

        printf("Prediction: %d | Truth: %d | %s\n",
               rf_pred, y_true, (rf_pred == y_true) ? "CORRECT" : "WRONG");
        printf("Latency: %.2f ns\n", rf_ns/1000);

        if (rf_pred == y_true) correct++;
        total++;

        fprintf(out, "%d,%.2f\n", it, rf_ns);
        fflush(out);

        sleep(1);   /* pause so the demo advances one sample per second */
    }

    double accuracy = (total > 0) ? ((double)correct / (double)total) : 0.0;
    printf("RF online accuracy: %.6f (%" PRIu64 "/%" PRIu64 ")\n",
           accuracy, correct, total);

    fclose(out);

    free_rf(rf);
    free(rf);

    return 0;
}