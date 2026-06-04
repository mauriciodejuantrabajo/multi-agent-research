"""
Capa de acceso al LLM (DeepSeek) con soporte de tool-calling.

La API de DeepSeek es compatible con el formato de OpenAI: el modelo puede
responder con `tool_calls` (pedidos estructurados para ejecutar una herramienta).
El agente los ejecuta y le devuelve el resultado, en un loop, hasta que el modelo
responde con texto final.

DeepSeek se configura por variables de entorno (ver .env.example):
    DEEPSEEK_API_KEY   tu API key (https://platform.deepseek.com)
    DEEPSEEK_MODEL     modelo a usar (por defecto: deepseek-v4-flash)
    DEEPSEEK_BASE_URL  endpoint (por defecto: https://api.deepseek.com)
"""

from __future__ import annotations

import os
from typing import Any

import requests


class LLMError(RuntimeError):
    """Error al comunicarse con el backend del LLM."""


class DeepSeekClient:
    """Cliente para la API de DeepSeek (compatible OpenAI) con function-calling."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 180,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.base_url = (
            base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        ).rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Envía la conversación y devuelve el mensaje del asistente.

        El mensaje puede contener `content` (texto) y/o `tool_calls` (lista de
        herramientas que el modelo quiere ejecutar).
        """
        if not self.api_key:
            raise LLMError(
                "Falta DEEPSEEK_API_KEY. Copiá .env.example a .env y poné tu key "
                "(https://platform.deepseek.com)."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.0,
        }
        if tools:
            payload["tools"] = tools

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(
                f"No se pudo conectar a DeepSeek en {self.base_url}. "
                "¿Hay conexión a internet?"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            detail = ""
            try:
                detail = f" — {resp.json()}"
            except Exception:  # noqa: BLE001 — solo enriquecemos el mensaje de error
                pass
            raise LLMError(f"DeepSeek respondió con error: {exc}{detail}") from exc

        data = resp.json()
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Respuesta inesperada de DeepSeek: {data}") from exc


def get_client() -> DeepSeekClient:
    """Devuelve el cliente LLM configurado por variables de entorno."""
    return DeepSeekClient()
