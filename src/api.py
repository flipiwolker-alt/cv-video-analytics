import os
import time
import traceback
import base64
import tempfile
import subprocess
import httpx
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CV Video Analytics API")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
PROMPT = "List all objects visible in this image, one line, comma separated. Only the list, no explanations."


@app.get("/")
def root():
    return {"status": "ok", "api_key_loaded": bool(OPENROUTER_API_KEY)}


async def _describe_image_bytes(img_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                    json={
                        "model": "nvidia/nemotron-nano-12b-v2-vl:free",
                        "messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": PROMPT}
                        ]}]
                    }
                )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {traceback.format_exc()}")
            if attempt < 2:
                wait = 60 if "429" in str(e) else 10
                time.sleep(wait)
            else:
                raise


@app.post("/describe")
async def describe(file: UploadFile = File(...)):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY не задан")
    contents = await file.read()
    try:
        text = await _describe_image_bytes(contents)
        return JSONResponse({"objects": text, "filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class VideoRequest(BaseModel):
    url: str
    interval_sec: int = 5


@app.post("/describe-video")
async def describe_video(req: VideoRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY не задан")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")

        # download video via yt-dlp (handles YouTube, direct links, etc.)
        try:
            result = subprocess.run(
                ["yt-dlp", "-f", "mp4/best[ext=mp4]/best", "-o", video_path, req.url],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Не удалось скачать видео: {e}")

        # extract frames every interval_sec seconds
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Не удалось открыть видео")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = int(fps * req.interval_sec)
        results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                timestamp_sec = int(frame_idx / fps)
                _, buf = cv2.imencode(".jpg", frame)
                try:
                    objects = await _describe_image_bytes(buf.tobytes())
                    results.append({"time_sec": timestamp_sec, "objects": objects})
                except Exception as e:
                    results.append({"time_sec": timestamp_sec, "error": str(e)})
            frame_idx += 1

        cap.release()

    return JSONResponse({"url": req.url, "frames_analyzed": len(results), "results": results})
