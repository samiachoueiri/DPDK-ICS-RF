# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

All build commands run **on the BlueField-3 DPU** (ARM64), not the x86 host. The environment must be set up first.

```bash
# One-time: configure DOCA/DPDK pkg-config paths and allocate hugepages
source dpdk_add.sh

# Build fault detection binary (root Makefile, produces build/rf_app)
make shared

# Build attack detection binary (separate Makefile)
cd online/attack && make shared

# Clean
make clean
```

The binary outputs to `build/rf_app`. Run it with DPDK EAL arguments:
```bash
./build/rf_app -l 0 -n 4 -a 03:00.0 -- -p 0x1 -P
```
This writes per-sample latencies to `latencies.csv` (`iter,rf_ns` columns).

**Switching models:** Edit the `RF_MODEL_JSON` define at the top of `lat_rf.c` before building:
```c
#define RF_MODEL_JSON  "rf_ws8_depth32_est25.json"
```

## Offline ML Pipeline

Jupyter notebooks in `offline/` train models and export them to JSON:
- `offline/fault_pipeline.ipynb` — fault detection
- `offline/cyberattack_pipeline_modbus.ipynb` — cyberattack detection

Training features (CSV) are in `offline/fault_training_features/` and `offline/attack_trained_features/`, named `normal_ws{N}.csv` / `fault_ws{N}.csv` for window sizes N ∈ {4, 8, 16, 32}.

Trained models export to `offline/fault_trained_models_json/` as `rf_ws{N}_depth{D}_est{E}.json`.

## Architecture

The system has two independent pipelines (fault, attack) that share the same C architecture:

**Offline (Python, x86):**
1. Raw Modbus/TCP pcap → sliding-window feature extraction → CSV files
2. scikit-learn `RandomForestClassifier` grid search over `(n_estimators, max_depth, window_size)`
3. Trained model serialized to JSON using a custom exporter (stores `n_estimators`, `max_depth`, `feature_importances`, and per-tree arrays: `children_left`, `children_right`, `feature`, `threshold`, `class_label`, `leaves`, `n_nodes`)

**Online (C/DPDK, BlueField-3 ARM64):**
1. `rte_eal_init` initializes DPDK
2. `load_rf_model()` — reads the JSON model with Jansson; allocates each tree's `TreeNode` array on the heap (not stack) to avoid stack overflow on the DPU. Hard caps: `MAX_ALLOWED_ESTIMATORS=200`, `MAX_ALLOWED_NODES_PER_TREE=200000`, `MAX_TOTAL_NODES=2000000`.
3. `predict_tree()` — recursive tree traversal with bounds guard on node index
4. `predict_rf()` — majority-vote across all estimators
5. Benchmark loop over `X_test` from `test_data.h` (27,533 × 8 float samples); latency measured with `rte_rdtsc_precise()` and converted to nanoseconds via `rte_get_tsc_hz()`; accuracy computed inline against `y_expected_str[]`

**Key data structures:**
- `TreeNode` — flat array per tree: `left_child`, `right_child`, `feature`, `threshold`, `is_leaf`, `class_label`
- `RandomForest` — `n_estimators`, `max_depth`, `feature_importances[8]`, `TreeNode **trees`, `int *tree_node_counts`

**The 8 features** (fixed, `NUM_FEATURES=8`): `min`, `max`, `mean`, `range`, `slope`, `iat_min_us`, `iat_max_us`, `mean_iat_us` — extracted from a sliding window over Modbus/TCP register read responses.

## Source File Variants

Multiple `.c` variants exist for different measurement goals:
- `lat_rf.c` (root) — **latency measurement** on the DPU; writes `latencies.csv`
- `acc_mem_rf.c` — **accuracy + memory footprint** with granular `malloc`/`free` tracking
- `Lat_rf_generic.c` — generic latency variant
- `lat_rf_original.c` — original unsafe reference (stack allocation, kept for comparison)
- `online/attack/attack_rf.c` — same architecture, different model/features for attack detection
- `online/attack/attack_rf_demo.c` — demo variant

Only one `.c` file is compiled at a time (`SRCS-y := lat_rf.c` in Makefile).

## Transferring Files to DPU

The DPU is accessed at `192.168.100.2` via the host node as a jump proxy (see `online/fault/BF3_notes.txt` for the SSH config). Transfer workflow:
```bash
# From x86 host to DPU
scp -r *.c Makefile *.json test_data.h ubuntu@192.168.100.2:/home/ubuntu/rf
```

## Results Analysis

After running experiments, rename `latencies.csv` to `latencies_rf_ws8_depth32_est{N}.csv` and copy back to `results/fault/exp4_online/`. Visualization notebooks:
- `results/fault/figures.ipynb` — latency CDFs, accuracy/ROC-AUC heatmaps (Exp 1–4), SHAP plots
- `results/attack/cyberattack_figures.ipynb` — attack detection results
