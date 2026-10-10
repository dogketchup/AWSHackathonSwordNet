"""Minimal OpenAI-compatible client for the two agents (see config.LLMS):
    qwen    Qwen3-8B on our own Trainium chip, via the starter kit's vLLM server
    gptoss  the organisers' shared gpt-oss-20b endpoint
Same call the starter kit's agents make; see trainium-agent-labs/projects/*/agent.py.
"""
import json
import re

import httpx

from swordnet import config


def chat(prompt, which="qwen", max_tokens=2048, think=False, temperature=0.4):
    llm = config.LLMS[which]
    if not llm["base_url"]:
        raise RuntimeError(f"{which}: no endpoint configured (set it in .env)")
    body = dict(model=llm["model"], messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature, top_p=0.95)
    if which == "qwen":
        body["chat_template_kwargs"] = {"enable_thinking": think}
    r = httpx.post(f"{llm['base_url'].rstrip('/')}/chat/completions", json=body, timeout=900)
    r.raise_for_status()
    # gpt-oss puts its hidden reasoning in a separate field; we only want the answer.
    text = r.json()["choices"][0]["message"].get("content") or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_json(text):
    """The first JSON object in a reply, tolerating ```json fences and chatter around it."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(candidate)


def available(which="qwen"):
    url = config.LLMS[which]["base_url"]
    if not url:
        return False
    try:
        return httpx.get(f"{url.rstrip('/')}/models", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False
