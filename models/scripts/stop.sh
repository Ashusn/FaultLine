#!/usr/bin/env bash
pkill -f real.sh; pkill -f "run.py"; sleep 5
pkill -9 -f "run.py"; pkill -9 -f "multiprocessing.spawn"; pkill -9 -f EngineCore; sleep 10
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "start again only when the last line shows 0 MiB"
