# Deployment

## Local (CPU)
```bash
sudo apt-get install -y libegl1 libgles2            # needed by MediaPipe 1.x Tasks on Linux
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
pip install -r requirements.txt
bash scripts/download_models.sh                      # MediaPipe holistic_landmarker.task (13.7 MB)
python scripts/download_wlasl.py --out data/raw      # ~480 MB, WLASL100 subset (C-UDA, non-commercial)
python scripts/extract_landmarks.py --workers 4      # ~12 min on 4 vCPU for 1,007 clips (711 s measured)
python scripts/check_consistency.py                  # results/consistency.json
python scripts/run_experiments.py --seeds 0,1,2      # results/ablation.json
python scripts/train_best.py                         # artifacts/best/model.pt (V-best config, seed 0)
python scripts/demo.py --video data/raw/videos/<id>.mp4
pytest -q                                            # no dataset needed
```

## Webcam
`python scripts/demo.py --webcam 0 --show` needs a desktop OpenCV build (`pip install opencv-python` instead of the headless wheel)
and a display. Not tested in this environment (no camera, no display). The video-file path is tested.

## Docker
`docker build -t slr . && docker run --rm slr` runs the test suite. Mount data to run the pipeline:
`docker run --rm -v $PWD/data:/app/data slr python scripts/extract_landmarks.py`.

## Cost / latency
- Landmark extraction ≈ 55 ms per frame on one CPU core at ≤640 px (HolisticLandmarker float16). A 65-frame clip ≈ 3.6 s.
- The classifier itself is ~0.5 M parameters; inference cost is negligible next to landmark extraction.

## Not production-ready
- Isolated-sign classification over 100 glosses only; no continuous signing, no sentence-level recognition.
- Trained on ~740 clips; accuracy is far below what a usable product needs (see README).
- No signer-independent evaluation (31 of 34 test signers also appear in training).
