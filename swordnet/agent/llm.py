"""Minimal client for the Qwen3 server on the Trainium chip (vLLM, OpenAI-compatible).

Same call the starter kit's agents make; see trainium-agent-labs/projects/*/agent.py.
"""
import json
import re

import httpx

from swordnet import config


def chat(prompt, max_tokens=2048, think=False, temperature=0.4):
    body = dict(model=config.LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature, top_p=0.95,
                chat_template_kwargs={"enable_thinking": think})
    r = httpx.post(f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions", json=body, timeout=900)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"].get("content") or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_json(text):
    """The first JSON object in a reply, tolerating ```json fences and chatter around it."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(candidate)


def available():
    try:
        return httpx.get(f"{config.LLM_BASE_URL.rstrip('/')}/models", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False
