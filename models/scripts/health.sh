#!/usr/bin/env bash
date -u
ps aux | grep -E '[r]eal\.sh|[m]irror|[r]un\.py' | awk '{print $2, $12, $13, $14, $15}' | head
nvidia-smi --query-gpu=memory.used --format=csv,noheader
du -sh /workspace/* 2>/dev/null | sort -h | tail -n 10
tail -c 3000 /workspace/lab_logs/night.log | tr '\r' '\n' | grep -v '%|' | grep -v '^INFO' | tail -n 20
