from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import anthropic
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Mount static files (the frontend HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load API key from environment variable (set in Render dashboard)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Load system prompt from file
def get_system_prompt():
    try:
        with open("system_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "You are a helpful academic advisor for the Busch School of Business at Catholic University of America."

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.get("/health")
async def health():
    return {"status": "ok", "key_configured": bool(ANTHROPIC_API_KEY)}

@app.post("/chat")
async def chat(request: Request):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured. Please set ANTHROPIC_API_KEY in your environment variables.")

    try:
        body = await request.json()
        messages = body.get("messages", [])

        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided.")

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        system_prompt = get_system_prompt()

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=messages
        )

        return {
            "content": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }

    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key. Please check your ANTHROPIC_API_KEY environment variable.")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit reached. Please try again in a moment.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
