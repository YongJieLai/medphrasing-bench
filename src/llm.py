"""统一 LLM 客户端：从 config/models.yaml 读取端点，key 从环境变量注入。"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent


def _load_config() -> dict:
    with open(ROOT / "config" / "models.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


_CFG = _load_config()
_clients: dict[str, OpenAI] = {}


def _client(provider: str) -> OpenAI:
    if provider not in _clients:
        p = _CFG["providers"][provider]
        key = os.getenv(p["api_key_env"])
        if not key:
            raise RuntimeError(
                f"缺少环境变量 {p['api_key_env']}，请复制 .env.example 为 .env 并填入 key"
            )
        _clients[provider] = OpenAI(api_key=key, base_url=p["base_url"])
    return _clients[provider]


def chat(role: str, messages: list[dict], temperature: float = 0.0,
         max_tokens: int = 1024, **kwargs) -> str:
    """以指定角色调用模型，返回文本。

    role: patient / doctor / extractor（见 config/models.yaml 的 roles）
    """
    r = _CFG["roles"][role]
    client = _client(r["provider"])
    resp = client.chat.completions.create(
        model=r["model"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return resp.choices[0].message.content or ""
