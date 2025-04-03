import re
from typing import Optional
import numpy as np
from ollama import Client


class LLM:
    def __init__(
        self,
        base_url: Optional[str] = "http://localhost:11434",
        llm_model: Optional[str] = "phi4:latest",
        embed_model: Optional[str] = "nomic-embed-text:latest",
    ):
        self.base_url = base_url
        self.llm_model = llm_model
        self.embed_model = embed_model

        self.ollama_client = Client(host=self.base_url)

    def chat(self, messages: list[dict[str, str]]) -> str:
        """[{"role": "user", "content": "Why is sky blue?"}]"""
        response = self.ollama_client.chat(
            model=self.llm_model,
            messages=messages,
            options={"max_tokens": 1000, "temperature": 0},
        ).message.content
        return response

    def embed(self, text_list: list[str]) -> dict[str, np.ndarray]:
        embeddings = self.ollama_client.embed(
            model=self.embed_model, input=text_list
        ).embeddings
        embeddings = {
            text: np.array(embedding) for text, embedding in zip(text_list, embeddings)
        }
        return embeddings

    def parse_response(
        self,
        response: str,
        pattern: str,
        columns: dict[str, type],
    ) -> list[dict[str, any]]:
        """
        >>> results = self._parse_llm_response(
        >>>     response,
        >>>     r"\{(.*?)\s*\(Score:\s*([0-9.]+)\)\s*\}",
        >>>     {"entity": str, "score": float},
        >>> )
        """
        # print(response)
        matches = re.findall(pattern, response)
        results = []
        for match in matches:
            if not isinstance(match, tuple):
                match = (match,)
            results.append(
                {column: item.strip() for column, item in zip(columns, match)}
            )

        for idx, result in enumerate(results):
            for key, value in result.items():
                results[idx][key] = columns[key](value)
        return results


if __name__ == "__main__":
    llm = LLM()

    response = llm.chat([{"role": "user", "content": "Why is sky blue?"}])
    print(response)

    embeddings = llm.embed(["hi", "Hi"])
    print(embeddings)
