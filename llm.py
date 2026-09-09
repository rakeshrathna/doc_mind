import json
import os
import re
import requests

HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def chat(system, user, temperature=0.0, json_mode=True):
    payload = {
        "model": MODEL,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        payload["format"] = "json"
    r = requests.post(f"{HOST}/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse_json(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise
        return json.loads(m.group(0))
