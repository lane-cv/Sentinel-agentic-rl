#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
mkdir -p ~/tunnel_logs
while true; do
    Sentinel-agent/.venv/bin/python verl_adapter/rag_service.py >> ~/tunnel_logs/rag_service.log 2>&1
    echo "[$(date)] rag_service exited ($?), restarting in 5s" >> ~/tunnel_logs/rag_service.log
    sleep 5
done
