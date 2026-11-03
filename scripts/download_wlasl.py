"""Download the WLASL100 subset (videos still available in the Voxel51 HF mirror).

Sources
  - Labels + official splits + signer ids: WLASL_v0.3.json from https://github.com/dxli94/WLASL (start_kit/)
  - Video files: Hugging Face dataset mirror https://huggingface.co/datasets/Voxel51/WLASL (data/data_*/<video_id>.mp4)
License: WLASL is released for academic, computational, non-commercial use under the C-UDA
(Computational Use of Data Agreement). Videos are NOT committed to this repo.

  python scripts/download_wlasl.py --out data/raw --glosses 100
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WLASL_JSON = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
HF_SAMPLES = "https://huggingface.co/datasets/Voxel51/WLASL/resolve/main/samples.json"
HF_FILE = "https://huggingface.co/datasets/Voxel51/WLASL/resolve/main/{path}"


def fetch(url: str, dest: Path, tries: int = 6) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wlasl-subset-downloader/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.rename(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:  # listed in the mirror's metadata but file absent: permanent
                print(f"  404 {url.rsplit('/', 1)[-1]} (skipped)")
                return False
            wait = 2 ** k
            print(f"  retry {k + 1}/{tries} {url.rsplit('/', 1)[-1]} in {wait}s: {e}")
            time.sleep(wait)
        except Exception as e:  # transient network errors: back off
            wait = 2 ** k
            print(f"  retry {k + 1}/{tries} {url.rsplit('/', 1)[-1]} in {wait}s: {e}")
            time.sleep(wait)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--glosses", type=int, default=100, help="first N glosses of WLASL_v0.3.json (100 = official WLASL100)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    meta_dir = a.out / "meta"
    fetch(WLASL_JSON, meta_dir / "WLASL_v0.3.json")
    fetch(HF_SAMPLES, meta_dir / "hf_samples.json")
    wlasl = json.loads((meta_dir / "WLASL_v0.3.json").read_text())
    hf = {Path(s["filepath"]).stem: s for s in json.loads((meta_dir / "hf_samples.json").read_text())["samples"]}

    rows, missing = [], 0
    for g in wlasl[: a.glosses]:
        for inst in g["instances"]:
            vid = inst["video_id"]
            if vid not in hf:
                missing += 1
                continue
            assert hf[vid]["gloss"]["label"] == g["gloss"], vid
            rows.append({"video_id": vid, "gloss": g["gloss"], "split": inst["split"], "signer_id": inst["signer_id"],
                         "frame_start": inst["frame_start"], "frame_end": inst["frame_end"], "fps": inst["fps"],
                         "hf_path": hf[vid]["filepath"], "file": f"videos/{vid}.mp4"})
    print(f"{len(rows)} videos available in mirror, {missing} listed in WLASL but not in mirror (dead links upstream)")

    def job(r):
        return fetch(HF_FILE.format(path=r["hf_path"]), a.out / r["file"])

    with ThreadPoolExecutor(a.workers) as ex:
        ok = list(ex.map(job, rows))
    rows = [r for r, k in zip(rows, ok) if k]
    (a.out / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"downloaded {sum(ok)}/{len(ok)}; manifest -> {a.out / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
