#!/usr/bin/env bash
set -euo pipefail

RENDERS_DIR="/home/ubuntu/fog/renders"
DATA_DIR="/home/ubuntu/fog/data"

echo "$(date -Iseconds) - cleaning old files in $RENDERS_DIR and $DATA_DIR"

# Delete files older than 60 minutes in renders/
find "$RENDERS_DIR" -type f -mmin +60 -print -delete

# Delete files older than 60 minutes in data/
find "$DATA_DIR" -type f -mmin +60 -print -delete