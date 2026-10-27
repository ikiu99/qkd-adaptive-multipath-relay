#!/bin/bash
# Copy the pool server into the ns-3 scratch folder and build it.
# Usage (inside WSL): bash build_server.sh [WORKDIR]
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
NS3=${1:-$HOME/qkdmp-ns3}/ns-3-dev
cp "$HERE/ns3/qkdmp-pool-server.cc" "$NS3/scratch/qkdmp-pool-server.cc"
cd "$NS3"
./ns3 build scratch/qkdmp-pool-server 2>&1 | grep -E "error|warning: unused|Finished|\[100%\]" || true
ls -la "$NS3/build/scratch/ns3.46-qkdmp-pool-server-optimized"
