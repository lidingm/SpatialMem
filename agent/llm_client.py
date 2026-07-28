"""OpenAI-compatible LLM client using httpx."""

from __future__ import annotations

import json
import re
import time

import httpx


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 300,
        max_retries: int = 3,
        temperature: float = 0.7,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    def chat(self, messages: list[dict], temperature: float | None = None,
             max_tokens: int = 8192,
             response_format: dict | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens,
        }
        if "qwen" in self.model.lower() or "dashscope" in self.model.lower():
            payload["enable_thinking"] = False  # Qwen3 thinking off: saves tokens and latency.
        if response_format is not None:
            payload["response_format"] = response_format
        data = self._make_request(payload)
        return data["choices"][0]["message"]["content"]

    def chat_json(self, messages: list[dict], temperature: float | None = None,
                  max_tokens: int = 8192, include_raw: bool = False) -> dict:
        msgs = list(messages)
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\nYou MUST respond with valid JSON only. No markdown fences, no explanation outside the JSON."}
        else:
            msgs.insert(0, {"role": "system", "content": "You MUST respond with valid JSON only."})

        # Try structured-output mode first; falls back to freeform if the
        # endpoint doesn't support response_format (some proxies don't).
        rf: dict | None = {"type": "json_object"}
        for attempt in range(3):
            try:
                raw = self.chat(msgs, temperature=temperature,
                                max_tokens=max_tokens, response_format=rf)
            except httpx.HTTPStatusError:
                rf = None
                raw = self.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            parsed = self._try_parse_json(raw)
            if parsed is not None:
                if include_raw:
                    parsed = dict(parsed)
                    parsed["_raw_response"] = raw
                return parsed
            msgs.append({"role": "assistant", "content": raw})
            msgs.append({"role": "user", "content": "Your response was not valid JSON. Please fix it and respond with valid JSON only."})
        raise ValueError(
            "Failed to get valid JSON after 3 attempts. "
            f"Full last response:\n{raw}"
        )

    def _make_request(self, payload: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(url, json=payload)
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(2 ** (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.TimeoutException:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise
        raise RuntimeError(f"LLM request failed after {self.max_retries} retries")

    @staticmethod
    def _extract_partial_json_fields(text: str) -> dict | None:
        answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', text, re.DOTALL)
        if not answer_match:
            return None

        result: dict = {"answer": answer_match.group(1)}

        reasoning_match = re.search(r'"reasoning_chain"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1)
            reasoning = reasoning.replace(r'\n', '\n').replace(r'\"', '"').replace(r'\\', '\\')
            result["reasoning_chain"] = reasoning
        else:
            result["reasoning_chain"] = ""

        conf_match = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                result["confidence"] = 0.0
        else:
            result["confidence"] = 0.0

        key_match = re.search(r'"key_evidence"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if key_match:
            items = re.findall(r'"((?:[^"\\]|\\.)*)"', key_match.group(1), re.DOTALL)
            result["key_evidence"] = [
                item.replace(r'\n', '\n').replace(r'\"', '"').replace(r'\\', '\\')
                for item in items
            ]
        else:
            result["key_evidence"] = []

        result["partial_parse"] = True
        return result

    @classmethod
    def _try_parse_json(cls, text: str) -> dict | None:
        text = text.strip()
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        # 尝试找到第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return cls._extract_partial_json_fields(text)
