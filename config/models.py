from typing import TypedDict


class ModelConfig(TypedDict):
    id: str
    description: str
    temperature: float
    max_tokens: int


MODELS: dict[str, ModelConfig] = {
    "gpt-4o-mini": {
        "id": "openai/gpt-4o-mini",
        "description": "GPT-4o Mini (OpenAI)",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "gpt-4o": {
        "id": "openai/gpt-4o",
        "description": "GPT-4o (OpenAI)",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "claude-sonnet": {
        "id": "anthropic/claude-sonnet-4-5",
        "description": "Claude Sonnet 4.5 (Anthropic)",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "gemini-flash": {
        "id": "vertex_ai/gemini-2.5-flash",
        "description": "Gemini 2.5 Flash (Google)",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "gemini-pro": {
        "id": "vertex_ai/gemini-2.5-pro",
        "description": "Gemini 2.5 Pro (Google)",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "llama3": {
        "id": "huggingface/meta-llama/Llama-3.3-70B-Instruct",
        "description": "Llama 3.3 70B (Meta)",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "gpt-5.4-mini": {
        "id": "openai/gpt-5.4-mini",
        "description": "GPT-5.4 Mini (OpenAI, 2026-03)",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "gemini-3.5-flash": {
        "id": "vertex_ai/gemini-3.5-flash",
        "description": "Gemini 3.5 Flash (Google, 2026-05)",
        "temperature": 0.0,
        "max_tokens": 8192,
    },
}


def resolve_model_key(key: str | None) -> str | None:
    """모델 키를 LiteLLM 모델 ID로 변환. 키가 MODELS에 없으면 원본 반환."""
    if key and key in MODELS:
        return MODELS[key]["id"]
    return key
