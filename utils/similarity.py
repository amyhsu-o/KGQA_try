import numpy as np
from rapidfuzz import fuzz
from lm.llm import LLM


def get_fuzzy_best_match(
    target: str, candidates: list[str], threshold: int = 80, top_n: int = 1
) -> dict[str, float]:
    best_matches_score = []
    for candidate in candidates:
        score = fuzz.ratio(target.lower(), candidate)
        if score >= threshold:
            best_matches_score.append((candidate, score))
    best_matches_score = sorted(best_matches_score, key=lambda x: x[1], reverse=True)[
        :top_n
    ]
    best_matches_score = {match: score for match, score in best_matches_score}
    return best_matches_score


def get_cosine_similarity_best_match(
    target_emb: np.ndarray,
    candidates_emb: dict[str, np.ndarray],
    threshold: int = 0.8,
    top_n: int = 1,
) -> dict[str, float]:
    best_matches_score = []
    for candidate, candidate_emb in candidates_emb.items():
        score = np.dot(target_emb, candidate_emb) / (
            np.linalg.norm(target_emb) * np.linalg.norm(candidate_emb)
        )
        if score >= threshold:
            best_matches_score.append((candidate, score))
    best_matches_score = sorted(best_matches_score, key=lambda x: x[1], reverse=True)[
        :top_n
    ]
    best_matches_score = {match: score for match, score in best_matches_score}
    return best_matches_score


if __name__ == "__main__":
    target = "apple"
    candidates = ["apple", "banana", "orange", "grape", "kiwi"]

    # fuzzy similarity
    print(f"fuzzy: {get_fuzzy_best_match(target, candidates)}")
    print(f"fuzzy: {get_fuzzy_best_match(target, candidates, threshold=0, top_n=3)}")

    # cosine similarity
    llm_model = LLM()
    target_emb = llm_model.embed([target])[target]
    candidates_emb = llm_model.embed(candidates)
    print(f"embedding: {get_cosine_similarity_best_match(target_emb, candidates_emb)}")
    print(
        f"embedding: {get_cosine_similarity_best_match(target_emb, candidates_emb, threshold=0, top_n=3)}"
    )
