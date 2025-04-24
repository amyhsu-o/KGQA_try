import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import os
import logging
from qa.QA import QA
from lm.llm import LLM
from utils.similarity import get_cosine_similarity_best_match


class NaiveRAG(QA):
    answer_prompt = """Given a question and some related document chunks, you are asked to answer the question with these contents and your knowledge."""

    def __init__(
        self,
        chunks: list[str],
        top_n: int = 3,
        llm_verbose: bool = False,
        logger: logging.Logger = logging.getLogger(),
    ):
        self.llm_model = LLM(verbose=llm_verbose, logger=logger)
        self._get_chunks_embedding(chunks)
        self.top_n = top_n
        self.logger = logger

    def _get_chunks_embedding(self, chunks: list[str]) -> None:
        self.chunks_emb = self.llm_model.embed(chunks)

    def retrieve(self, query: str) -> list[str]:
        query_emb = self.llm_model.embed([query])[query]
        best_matches = get_cosine_similarity_best_match(
            query_emb, self.chunks_emb, top_n=self.top_n
        )
        return [match for match in best_matches]

    def answer(self, query: str) -> str:
        related_chunks = "\n\n".join(self.retrieve(query))
        response = self.llm_model.chat(
            [
                {
                    "role": "system",
                    "content": NaiveRAG.answer_prompt,
                },
                {
                    "role": "user",
                    "content": f"Q: {query}\nRelated document chunks:\n{related_chunks}\nA:",
                },
            ]
        )
        return response


if __name__ == "__main__":
    pass
