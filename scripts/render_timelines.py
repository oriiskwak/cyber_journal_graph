#!/usr/bin/env python3
"""Render reviewed excerpts from the local source videos with saved TEMPO scores."""

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    {
        "name": "scam", "label": "사기 사례 · TP", "id": "abnormal_BU2FfjTqGY8",
        "hash": "10cdf22f73ee92e3cebdd6c08b896e00d34f8756f81b844cc61f6624bc7ef1a9",
        "seconds": range(13), "crop": "crop=455:120:165:925",
        "text": "24시간 VVIP 리딩방 초대",
        "objects": "채팅 UI · 메시지 목록 · 홍보 배너",
    },
    {
        "name": "normal", "label": "정상 사례 · TN", "id": "normal_qsHWozaXnnw",
        "hash": "005d881d2950dfae9fd9bd0ea7d63e6bc164f8731eb428b84c0ae4769ab2391e",
        "seconds": range(40), "crop": "crop=945:200:65:310",
        "text": "보이스피싱 수거책 / 50대에게 징역 2년 선고",
        "objects": "뉴스 제목 그래픽 · 자막 상자",
    },
)


def capture(video, second, crop):
    image = subprocess.check_output([
        "ffmpeg", "-v", "error", "-ss", str(second), "-i", str(video),
        "-vf", crop + ",scale=336:-2", "-frames:v", "1", "-map_metadata", "-1",
        "-f", "image2pipe", "-c:v", "png", "pipe:1",
    ])
    return Image.open(io.BytesIO(image)).convert("RGB")


def render_page(path, case, page_rows, images, font_path):
    ink, muted, accent = "#14243b", "#415873", "#2c63a6"
    fonts = {size: ImageFont.truetype(str(font_path), size) for size in (18, 20, 23, 34)}
    width, height = 1504, 184 + 235 * ((len(page_rows) + 3) // 4)
    page = Image.new("RGB", (width, height), "#f2f5fa")
    draw = ImageDraw.Draw(page)
    first, last = page_rows[0]["second"], page_rows[-1]["second"]
    draw.text((28, 22), f"{case['label']}  |  {first:02d}–{last:02d}초", font=fonts[34], fill=ink)
    draw.text((28, 76), f"{case['id']}  ·  사람 검토로 공개 범위를 정한 화면 발췌", font=fonts[20], fill=muted)
    draw.text((28, 108), f"화면 문구    {case['text']}", font=fonts[20], fill=ink)
    draw.text((28, 140), f"허용 객체    {case['objects']}  ·  사람의 사후 관찰", font=fonts[20], fill=muted)
    for i, row in enumerate(page_rows):
        x = 24 + (i % 4) * 368
        y = 178 + (i // 4) * 235
        draw.rounded_rectangle((x, y, x + 352, y + 216), radius=10, fill="white")
        draw.text((x + 8, y + 8), f"{row['second']:02d}초", font=fonts[23], fill=ink)
        shot = images[row["second"]]
        page.paste(shot, (x + 8, y + 43))
        score = row["model_score"]
        score_text = "저장된 추론 시점 아님" if score == "" else f"TEMPO {row['segment']}  ·  {score:.2f}"
        draw.text((x + 8, y + 155), score_text, font=fonts[20], fill=muted if score == "" else accent)
        cue = "객체별 판정 기여 기록 없음" if score != "" else "발표용 1초 캡처"
        draw.text((x + 8, y + 182), cue, font=fonts[18], fill=muted)
    page.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"))
    args = parser.parse_args()
    source = json.loads((ROOT / "run/presentation_cases.json").read_text())
    by_id = {row["video_id"]: row for row in source["videos"]}
    target = ROOT / "timeline"
    target.mkdir(exist_ok=True)
    for case in CASES:
        video = args.video_dir / f"{case['id']}.mp4"
        if hashlib.sha256(video.read_bytes()).hexdigest() != case["hash"]:
            raise ValueError(f"Video differs from the reviewed source: {video.name}")
        scores = {row["second"]: row for row in by_id[case["id"]]["frames"]}
        images = {second: capture(video, second, case["crop"]) for second in case["seconds"]}
        rows = []
        for second in case["seconds"]:
            frame = scores.get(second, {})
            rows.append({"video_id": case["id"], "second": second, "screen_text_excerpt": case["text"],
                         "reviewed_object_allowlist": case["objects"], "segment": frame.get("segment", ""),
                         "model_score": frame.get("model_score", ""),
                         "observed_content_used_as_structured_classifier_input": False,
                         "object_level_attribution_recorded": False})
        with (target / f"{case['name']}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        for start in range(0, len(rows), 8):
            render_page(target / f"{case['name']}_{start:02d}.png", case, rows[start:start + 8], images, args.font)


if __name__ == "__main__":
    main()
