#!/bin/bash
# Build a clean, isolated ns-3.46 + QKDNetSim tree for the prototype.
#
# Sources are cloned from local git repositories (no network needed) so the
# tree is bit-identical to upstream ns-3.46 and QKDNetSim master, without the
# uncommitted changes of other projects living in the source trees.
#
# Usage (inside WSL):  bash setup_ns3.sh [SRC_NS3] [WORKDIR]
set -euo pipefail

SRC_NS3=${1:-$HOME/qkd-project/ns-3-dev}
WORK=${2:-$HOME/qkdmp-ns3}
NS3=$WORK/ns-3-dev

mkdir -p "$WORK"
if [ ! -d "$NS3/.git" ]; then
  git clone --quiet --no-hardlinks "$SRC_NS3" "$NS3"
  git -C "$NS3" checkout --quiet ns-3.46
fi
if [ ! -d "$NS3/contrib/qkdnetsim/.git" ]; then
  git clone --quiet --no-hardlinks "$SRC_NS3/contrib/qkdnetsim" "$NS3/contrib/qkdnetsim"
fi

cd "$NS3"
# Bug fix for QKDNetSim: QKDKMSQueueLogic's constructor reads NumberOfQueues
# before ns-3 sets attributes (uninitialised -> random, often huge, allocation).
PATCH="$(cd "$(dirname "$0")/.." && pwd)/ns3/qkdnetsim-queue-init-fix.patch"
if git -C contrib/qkdnetsim apply --check "$PATCH" 2>/dev/null; then
  git -C contrib/qkdnetsim apply "$PATCH"
fi
# gnuplot patches required by QKDNetSim (see its README); skip if already applied.
for p in gnuplot_h gnuplot_cc; do
  if git apply --check "contrib/qkdnetsim/patches/$p.patches" 2>/dev/null; then
    git apply "contrib/qkdnetsim/patches/$p.patches"
  fi
done

# QKDNetSim includes aodv/dsdv/olsr headers without declaring them as
# dependencies, so they are enabled explicitly.
./ns3 configure -d optimized --enable-modules "qkdnetsim;aodv;dsdv;olsr;point-to-point;traffic-control;applications;internet;stats" --disable-tests --disable-examples --disable-python > "$WORK/configure.log" 2>&1
./ns3 build > "$WORK/build.log" 2>&1
echo "BUILD OK" >> "$WORK/build.log"
