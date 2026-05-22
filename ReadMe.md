# DPDK-ICS-RF

**In-network fault and cyberattack detection for Industrial Control Systems (ICS) using a Random Forest classifier deployed directly on an NVIDIA BlueField-3 SmartNIC via DPDK.**

Instead of forwarding ICS traffic to a server for analysis, this system runs the entire ML inference pipeline — feature extraction and Random Forest prediction — inside the SmartNIC's data plane. The result is sub-microsecond detection with zero impact on host CPU resources.

---

## How It Works

The system is split into two phases:

```
Offline (x86 host)                       Online (BlueField-3 DPU)
─────────────────────────────────────    ──────────────────────────────────────
  Modbus/TCP pcap traces                   Live Modbus/TCP traffic
         │                                          │
  Sliding-window feature                    DPDK packet receive (rte_eal)
  extraction (ws=4/8/16/32)                         │
         │                                  Flow-level feature extraction
  scikit-learn RF training                  (same 8 features, same window)
  (grid search over depth,                          │
   n_estimators, window size)               JSON model loaded at startup
         │                                          │
  Model exported to JSON                    RF inference (C, ARM NEON)
  (rf_ws8_depth32_est*.json)                        │
                                            Normal / Fault / Attack label
                                            + per-sample latency timestamp
```

Both **fault detection** (sensor/equipment failures) and **cyberattack detection** (ICS-targeted intrusions on Modbus/TCP) are supported as separate pipelines.

---

## Dataset & Feature Engineering

**Dataset:** [ICS Dataset — Sudyana (Zenodo, 2025)](https://doi.org/10.5281/zenodo.17165850)

Traffic is captured from a Modbus/TCP-based cyber-physical system under both normal operation and anomalous conditions (faults and attacks). A sliding window over each flow produces 8 statistical features per sample:

| Feature | Description |
|---|---|
| `min` | Minimum register value in window |
| `max` | Maximum register value in window |
| `mean` | Mean register value |
| `range` | Max − min |
| `slope` | Linear trend across the window |
| `iat_min_us` | Minimum inter-arrival time (μs) |
| `iat_max_us` | Maximum inter-arrival time (μs) |
| `mean_iat_us` | Mean inter-arrival time (μs) |

**SHAP analysis** shows that timing features dominate: `mean_iat_us` carries the highest predictive weight (mean |SHAP| = 0.139), followed by `iat_min_us` (0.101) and `iat_max_us` (0.095). Value-based features (`min`, `max`, `mean`) contribute but are secondary — anomalies manifest primarily in communication timing, not just register values.

Window sizes of 4, 8, 16, and 32 packets were evaluated. Larger windows improve accuracy (ws32 peaks at 89.1%) but increase buffering latency.

---

## Model & Hardware

**Model:** Random Forest classifier trained with scikit-learn, exported to a compact JSON format that the DPDK application loads at startup. Five model variants are included (1, 5, 10, 25, 50 trees; max depth 32 for fault, 8 for attack), covering the accuracy–latency–memory trade-off spectrum.

**Hardware:** NVIDIA BlueField-3 DPU
- ARM Cortex-A78 cores (ARM64)
- 200 Gb/s network interface
- Dedicated ARM NEON SIMD used in the inference hot path
- Provisioned via the [FABRIC testbed](fabric/slice.ipynb) for experiments

The C implementation uses heap-allocated tree arrays to prevent stack overflow on the DPU's constrained memory, with hard caps on total nodes (`MAX_TOTAL_NODES = 2,000,000`) to guard against runaway model sizes.

---

## Setup & Build

### 1. Environment (BlueField-3 DPU)

The DPU runs DOCA/DPDK. Set up pkg-config paths and hugepages:

```bash
source dpdk_add.sh
```

This script configures Mellanox DOCA, DPDK, and FlexIO paths for ARM64 and allocates hugepages.

### 2. Dependencies

- DPDK (tested with DOCA on BlueField-3)
- [Jansson](https://digip.org/jansson/) — JSON parsing for model loading (`-ljansson`)
- zlib (`-lz`)

### 3. Build

```bash
make shared   # or: make static
```

The default build target produces the latency-measurement binary (`lat_rf`). The attack detection variant lives in [`online/attack/`](online/attack/) with its own Makefile.

### 4. Run

```bash
./lat_rf -- [dpdk-eal-args]
```

To switch models, edit the `RF_MODEL_JSON` define in [`lat_rf.c`](lat_rf.c) before building:

```c
#define RF_MODEL_JSON  "rf_ws8_depth32_est25.json"
```

Pre-trained models available: `est1`, `est5`, `est10`, `est25`, `est50`.

### 5. Offline Training

Reproduce the training pipeline in [`offline/fault_pipeline.ipynb`](offline/fault_pipeline.ipynb) (fault) or [`offline/cyberattack_pipeline_modbus.ipynb`](offline/cyberattack_pipeline_modbus.ipynb) (attack). Outputs JSON models and CSV feature files compatible with the online component.

---

## Project Layout

```
DPDK-ICS-RF/
├── lat_rf.c                     # Main DPDK inference app (fault detection)
├── test_data.h                  # 27,533-sample test set (8 features)
├── rf_ws8_depth32_est*.json     # Pre-trained RF models (1–50 trees)
├── Makefile                     # DPDK build
├── dpdk_add.sh                  # DPU environment setup
├── offline/
│   ├── fault_pipeline.ipynb     # Fault detection training pipeline
│   ├── cyberattack_pipeline_modbus.ipynb
│   └── fault_trained_models_json/
├── online/
│   ├── fault/                   # Fault detection DPDK variants
│   └── attack/                  # Cyberattack detection DPDK app
├── results/
│   ├── fault/figures.ipynb      # Latency CDFs, accuracy heatmaps, SHAP
│   └── attack/
├── fabric/
│   └── slice.ipynb              # FABRIC testbed + BlueField-3 provisioning
└── data/
    └── link to data.txt         # Zenodo dataset DOI
```

---

## Citation

If you use the dataset:

```bibtex
@dataset{sudyana_2025_17165850,
  author    = {Sudyana, Didik},
  title     = {{ICS Dataset}},
  month     = sep,
  year      = 2025,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17165850},
  url       = {https://doi.org/10.5281/zenodo.17165850}
}
```
