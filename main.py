from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PROMPT_FILE = os.path.join(BASE_DIR, "system_prompt.txt")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def get_system_prompt():
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "You are a helpful academic advisor for the Busch School of Business at Catholic University of America."

SYSTEM_PROMPT = get_system_prompt()

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "key_configured": bool(GROQ_API_KEY),
        "prompt_chars": len(SYSTEM_PROMPT),
        "static_dir_exists": os.path.exists(STATIC_DIR),
    }

@app.post("/chat")
async def chat(request: Request):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured. Please set GROQ_API_KEY in Render environment variables.")

    try:
        body = await request.json()
        messages = body.get("messages", [])

        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided.")

        # Groq only supports text — strip image/document blocks
        cleaned_messages = []
        for msg in messages:
            if isinstance(msg["content"], str):
                cleaned_messages.append(msg)
            elif isinstance(msg["content"], list):
                text_parts = [b["text"] for b in msg["content"] if b.get("type") == "text"]
                if text_parts:
                    cleaned_messages.append({"role": msg["role"], "content": "\n".join(text_parts)})

        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *cleaned_messages
            ]
        )

        return {"content": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
