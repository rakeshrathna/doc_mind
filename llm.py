import json
import os
import re
import requests
from exceptions import OllamaUnavailableError, ModelNotFoundError, LLMParseError

HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def check_ollama_status(host=None, model=None):
    target_host = host or HOST
    target_model = model or MODEL
    try:
        r = requests.get(f"{target_host}/api/tags", timeout=5)
        r.raise_for_status()
        tags_data = r.json()
    except requests.exceptions.RequestException as e:
        raise OllamaUnavailableError(
            f"Cannot connect to Ollama host at {target_host}. Ensure Ollama is running."
        ) from e

    models = [m.get("name") for m in tags_data.get("models", []) if isinstance(m, dict)]
    # Match exact name or base model name (e.g., 'llama3.1:8b' or 'llama3.1:latest')
    if not any(target_model in m or m.startswith(target_model.split(":")[0]) for m in models):
        raise ModelNotFoundError(target_model)
    return True


def chat(system, user, temperature=0.0, json_mode=True):
    target_host = os.getenv("OLLAMA_HOST", HOST)
    target_model = os.getenv("OLLAMA_MODEL", MODEL)
    payload = {
        "model": target_model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        payload["format"] = "json"
    try:
        r = requests.post(f"{target_host}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.ConnectionError as e:
        raise OllamaUnavailableError(f"Failed to connect to Ollama service at {target_host}.") from e
    except requests.exceptions.HTTPError as e:
        if r.status_code == 404:
            raise ModelNotFoundError(target_model) from e
        raise RuntimeError(f"Ollama API HTTP error: {r.status_code}") from e
    except Exception as e:
        raise RuntimeError(f"LLM communication error: {str(e)}") from e


def extract_json_objects(text):
    """Find candidate JSON substrings using balanced brace matching."""
    candidates = []
    stack = []
    start_idx = -1

    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_idx != -1:
                    candidates.append(text[start_idx:i + 1])
                    start_idx = -1
    return candidates


def parse_json(raw):
    if not raw or not isinstance(raw, str):
        raise LLMParseError("Raw LLM output is empty or non-string.")

    cleaned = raw.strip()

    # Case 1: Direct JSON parsing
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Case 2: Markdown code block stripping
    if "```" in cleaned:
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

    # Case 3: Balanced brace extraction for JSON embedded in commentary
    candidates = extract_json_objects(cleaned)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Case 4: Regex fallback for unclosed or unescaped single object
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise LLMParseError(f"Could not parse valid JSON from response: {raw[:200]}...")
