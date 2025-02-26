#!/bin/bash

echo "=== CPU Information ==="

# Number of physical cores
physical_cores=$(sysctl -n hw.physicalcpu)
echo "Number of physical cores: $physical_cores"

# Cache sizes
echo "=== Cache Sizes ==="
l1d_cache=$(sysctl -n hw.l1dcachesize)
l1i_cache=$(sysctl -n hw.l1icachesize)
l2_cache=$(sysctl -n hw.l2cachesize)
l3_cache=$(sysctl -n hw.l3cachesize)

echo "L1 Data Cache: $((l1d_cache / 1024)) KB"
echo "L1 Instruction Cache: $((l1i_cache / 1024)) KB"
echo "L2 Cache: $((l2_cache / 1024)) KB"
echo "L3 Cache: $((l3_cache / 1024)) KB"
