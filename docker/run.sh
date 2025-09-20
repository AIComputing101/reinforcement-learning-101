#!/usr/bin/env bash
set -euo pipefail

case "${1:-cpu}" in
  cpu)
    docker compose -f docker/docker-compose.yml up --build drl-cpu
    ;;
  cuda)
    docker compose -f docker/docker-compose.yml up --build drl-cuda
    ;;
  rocm)
    docker compose -f docker/docker-compose.yml up --build drl-rocm
    ;;
  *)
    echo "Usage: docker/run.sh [cpu|cuda|rocm]";
    exit 1;
    ;;
 esac
