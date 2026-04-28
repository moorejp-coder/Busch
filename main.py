from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os

# Absolute paths so Render can always find files
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

# Load API key from Render environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Load system prompt once at startup
def get_system_prompt():
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "You are a helpful academic advisor for the Busch School of Business at Catholic University of America."

SYSTEM_PROMPT = get_system_prompt()

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "key_configured": bool(GEMINI_API_KEY),
        "prompt_loaded": len(SYSTEM_PROMPT) > 100,
        "static_dir_exists": os.path.exists(STATIC_DIR),
    }

@app.post("/chat")
async def chat(request: Request):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API key not configured. Please set GEMINI_API_KEY in your Render environment variables."
        )

    try:
        body = await request.json()
        messages = body.get("messages", [])

        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided.")

        # Convert messages to Gemini format
        gemini_history = []
        for msg in messages[:-1]:  # All but the last message go into history
            content = msg["content"]
            if isinstance(content, list):
                # Extract text only
                text = "\n".join(
                    block["text"] for block in content
                    if block.get("type") == "text"
                )
            else:
                text = content

            gemini_history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [text]
            })

        # Get the last user message
        last_msg = messages[-1]
        last_content = last_msg["content"]
        if isinstance(last_content, list):
            last_text = "\n".join(
                block["text"] for block in last_content
                if block.get("type") == "text"
            )
        else:
            last_text = last_content

        # Create model with system prompt
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",  # Free tier, fast, large context window
            system_instruction=SYSTEM_PROMPT
        )

        # Start chat with history
        chat_session = model.start_chat(history=gemini_history)

        # Send the latest message
        response = chat_session.send_message(last_text)

        return {
            "content": response.text,
            "usage": {}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
