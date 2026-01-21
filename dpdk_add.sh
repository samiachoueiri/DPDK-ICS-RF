#!/bin/sh

# Add Mellanox DOCA pkg-config path
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:/opt/mellanox/doca/lib/aarch64-linux-gnu/pkgconfig

# Add Mellanox DOCA tools to PATH
export PATH=${PATH}:/opt/mellanox/doca/tools

# Add Mellanox DPDK pkg-config path
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:/opt/mellanox/dpdk/lib/aarch64-linux-gnu/pkgconfig

# Add Mellanox FlexIO pkg-config path
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:/opt/mellanox/flexio/lib/pkgconfig

# Add DPDK include pkg-config path
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:/opt/mellanox/dpdk/include/aarch64-linux-gnu/dpdk

sudo sh -c  "echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages"

echo "Environment configured."
