import os
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI(title="CV Video Analytics API")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
else:
    model = None

PROMPT = "Перечисли объекты на фото одной строкой через запятую. Только список, без пояснений."

@app.get("/")
def root():
    return {"status": "ok", "gemini_key_loaded": bool(GEMINI_API_KEY)}

@app.post("/describe")
async def describe(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY не задан")
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    resp = model.generate_content([img, PROMPT])
    return JSONResponse({"objects": resp.text.strip(), "filename": file.filename})
