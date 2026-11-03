#!/usr/bin/env bash
# MediaPipe Tasks model files (Apache-2.0), from Google's public model bucket.
set -euo pipefail
mkdir -p models
base=https://storage.googleapis.com/mediapipe-models
curl -sSfL -o models/holistic_landmarker.task $base/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
echo "e2dab61191e2dcd0  (sha256 prefix recorded 2026-11-03)"; sha256sum models/holistic_landmarker.task
