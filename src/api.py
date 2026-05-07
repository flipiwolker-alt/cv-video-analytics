import os
import time
import traceback
import base64
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CV Video Analytics API")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
PROMPT = "Перечисли объекты на фото одной строкой через запятую. Только список, без пояснений."

@app.get("/")
def root():
    return {"status": "ok", "api_key_loaded": bool(OPENROUTER_API_KEY)}

@app.post("/describe")
async def describe(file: UploadFile = File(...)):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY не задан")
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
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
                        "model": "google/gemini-2.0-flash-exp:free",
                        "messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": PROMPT}
                        ]}]
                    }
                )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return JSONResponse({"objects": text, "filename": file.filename})
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {traceback.format_exc()}")
            if attempt < 2:
                time.sleep(10)
            else:
                raise HTTPException(status_code=500, detail=str(e))
