#!/bin/bash
cd "$(dirname "$0")"
echo "Starting calibration tool at http://localhost:8000/homography_calibration.html"
echo "Press Ctrl+C to stop"
python3 -m http.server 8000 &
sleep 2  
xdg-open http://localhost:8000/homography_calibration.html
wait
