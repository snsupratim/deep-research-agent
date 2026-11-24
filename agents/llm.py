import os
import asyncio
from dotenv import load_dotenv
import httpx

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
# Set LLM_PROVIDER in .env to "groq" or "gemini" (defaults to "groq" if key present)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "gemini").lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


# ── Provider 1: Groq Engine ───────────────────────────────────────────────────
async def _call_groq(system: str, user: str, max_tokens: int, timeout: float) -> str:
    """Execute chat completion using Groq's high-speed API."""
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    model = os.getenv("GROQ_MODEL", GROQ_MODEL)

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing in .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")


# ── Provider 2: Google Gemini Engine ──────────────────────────────────────────
async def _call_gemini(system: str, user: str, max_tokens: int, timeout: float) -> str:
    """Execute chat completion using Google Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
    model_name = os.getenv("GEMINI_MODEL", GEMINI_MODEL)

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing in .env")

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    prompt = f"System Instructions:\n{system}\n\nUser Query:\n{user}"
    config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=0.3,
    )
    
    response = await asyncio.wait_for(
        asyncio.to_thread(model.generate_content, prompt, generation_config=config),
        timeout=timeout,
    )
    return response.text or ""


# ── Unified Multi-Model LLM Entrypoint ─────────────────────────────────────────
async def _llm(system: str, user: str, max_tokens: int = 2000, timeout: float = 60.0) -> str:
    """
    Unified LLM router supporting both Groq and Google Gemini families.
    Automatically switches based on LLM_PROVIDER in .env, with failover protection.
    """
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()

    if provider == "groq":
        try:
            return await _call_groq(system, user, max_tokens, timeout)
        except Exception as groq_err:
            print(f"⚠️ Groq attempt failed ({groq_err}).")
            if GEMINI_API_KEY:
                print("🔄 Falling back to Google Gemini...")
                return await _call_gemini(system, user, max_tokens, timeout)
            raise groq_err
    else:  # provider == "gemini"
        try:
            return await _call_gemini(system, user, max_tokens, timeout)
        except Exception as gemini_err:
            print(f"⚠️ Gemini attempt failed ({gemini_err}).")
            if GROQ_API_KEY:
                print("🔄 Falling back to Groq...")
                return await _call_groq(system, user, max_tokens, timeout)
            raise gemini_err
