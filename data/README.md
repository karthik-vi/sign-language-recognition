# data/

Nothing here is committed. Recreate with:

```bash
python scripts/download_wlasl.py --out data/raw --glosses 100     # ~480 MB, WLASL100 clips available in the HF mirror
bash scripts/download_models.sh                                    # MediaPipe holistic model
python scripts/extract_landmarks.py --raw data/raw --out data/landmarks --workers 4
```
See `docs/DATA.md` for source, license (C-UDA, non-commercial) and counts.
