# DPDK-ICS-RF

SmartNIC-accelerated Random Forest fault detection for Modbus/TCP cyber-physical systems.

This project offloads flow-level feature extraction and a pre-trained Random Forest model to an NVIDIA BlueField-3 SmartNIC using DPDK, enabling low-latency, in-network fault detection for ICS traffic.

## Key Features
- DPDK-based in-dataplane feature extraction
- Random Forest inference on SmartNIC
- Accuracy, ROC-AUC, latency, and memory footprint evaluation
- Hardware-aware model and feature selection
