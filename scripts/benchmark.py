"""Прогон метрик по списку видео-ссылок (GPU).

Пример:
    python scripts/benchmark.py --preset balanced ^
        --url "https://www.youtube.com/watch?v=XXXX" ^
        --url "https://www.youtube.com/watch?v=YYYY"

По каждому ролику печатает: длительность, время обработки, efficiency,
число событий по категориям и среднюю уверенность. В конце — markdown-таблица,
которую можно вставить в README.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import VideoAnalyzer       # noqa: E402
from src.device import describe              # noqa: E402


async def _one(url: str, preset: str, use_llm: bool, quality: str) -> dict:
    analyzer = VideoAnalyzer(
        dummy=False, offline=False, preset=preset, use_llm=use_llm,
    )
    t0 = time.time()
    report, _ = await analyzer.run(url, quality=quality)
    wall = time.time() - t0

    by_cls = Counter(d.subclass.value for d in report.detections)
    confs = defaultdict(list)
    for d in report.detections:
        confs[d.subclass.value].append(d.confidence)

    return {
        "url": url,
        "duration": report.sourceInfo.video_duration_seconds,
        "proc_time": report.sourceInfo.processing_time_seconds,
        "efficiency": report.sourceInfo.processing_efficiency,
        "wall": round(wall, 1),
        "n_det": len(report.detections),
        "by_cls": dict(by_cls),
        "mean_conf": {c: round(statistics.mean(v), 3) for c, v in confs.items()},
        "summary": report.llm_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", default=[], required=True,
                    help="ссылка на видео (можно несколько раз)")
    ap.add_argument("--preset", default="balanced",
                    choices=["fast", "balanced", "accurate"])
    ap.add_argument("--quality", default="480p")
    ap.add_argument("--no-llm", action="store_true", help="без LLM-слоя")
    args = ap.parse_args()

    print(f"Устройство: {describe()}")
    print(f"Пресет: {args.preset} | качество: {args.quality} | роликов: {len(args.url)}\n")

    rows = []
    for i, url in enumerate(args.url, 1):
        print(f"[{i}/{len(args.url)}] {url}")
        try:
            r = asyncio.run(_one(url, args.preset, not args.no_llm, args.quality))
        except Exception as exc:
            print(f"   ОШИБКА: {exc}\n")
            continue
        rows.append(r)
        print(f"   длительность {r['duration']:.0f}с | обработка {r['proc_time']:.1f}с "
              f"| efficiency {r['efficiency']} | событий {r['n_det']}")
        print(f"   по категориям: {r['by_cls']}")
        print(f"   ср. уверенность: {r['mean_conf']}\n")

    # ── Markdown-таблица для README ───────────────────────────────────────────
    if rows:
        print("\n=== Таблица для README ===\n")
        print("| Ролик | Длит. | Обработка | Efficiency | Событий | Категории (кол-во) |")
        print("|---|---|---|---|---|---|")
        for r in rows:
            cats = ", ".join(f"{k}: {v}" for k, v in sorted(r["by_cls"].items())) or "—"
            print(f"| {r['url']} | {r['duration']:.0f}с | {r['proc_time']:.1f}с "
                  f"| {r['efficiency']} | {r['n_det']} | {cats} |")


if __name__ == "__main__":
    main()
