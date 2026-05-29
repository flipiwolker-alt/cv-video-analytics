"""Разовый прогон пайплайна по URL — для проверки точности детекций.

    python scripts/run_once.py <youtube_url> [preset]
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import VideoAnalyzer  # noqa: E402


async def main():
    url = sys.argv[1]
    preset = sys.argv[2] if len(sys.argv) > 2 else "fast"

    async def on_progress(p, msg):
        print(f"[{p:5.0%}] {msg}", flush=True)

    analyzer = VideoAnalyzer(
        dummy=False, offline=False, preset=preset,
        use_ocr=False, use_clip=True, use_llm=True,
    )
    t0 = time.time()
    report, media = await analyzer.run(url, quality="480p", on_progress=on_progress)
    dt = time.time() - t0

    data = report.model_dump(mode="json")
    out = Path("outputs/run_once_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== РЕЗУЛЬТАТ ===", flush=True)
    print(f"preset={data.get('preset')}  время={dt:.1f}с  детекций={len(data['detections'])}", flush=True)
    by = {}
    for d in data["detections"]:
        by.setdefault(d["subclass"], []).append(d)
    for sub, ds in sorted(by.items(), key=lambda x: -len(x[1])):
        confs = [x["confidence"] for x in ds]
        print(f"  {sub:12s} x{len(ds):2d}  conf {min(confs):.2f}-{max(confs):.2f}  "
              f"типы={sorted(set(x['type'] for x in ds))}", flush=True)
    if data.get("llm_summary"):
        print("\nLLM summary:", data["llm_summary"], flush=True)
    print(f"\nJSON → {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
