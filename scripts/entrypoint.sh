#!/bin/sh
set -e

echo "Starting backend..."

python app.py &


if [ "$DEV_MODE" = "true" ]; then
  echo "DEV_MODE enabled: starting file watcher"
  python scripts/watch_and_process.py &
else
  echo "DEV_MODE disabled: watcher not running"
fi

while true; do
  echo "Sleeping 5 hours before clearing cache..."
  sleep 18000
  sh scripts/clear_cache.sh
done
