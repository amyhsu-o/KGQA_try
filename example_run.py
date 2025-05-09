import argparse
import os
import json
import logging
import threading
from data_loader import MuSiQueDataLoader
from construction import NERConstructor
from lm import LLM
from graph import KG
from qa import NaiveRAG, ToG, FastToG


def get_thread_logger(name: str, log_path: str, mode: str = "w") -> logging.Logger:
    logger = logging.getLogger(f"{name}_{threading.get_ident()}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode=mode)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def run_single_query(query_id: int, method: str, output_dir: str):
    logger = get_thread_logger(
        f"q{query_id}", f"{output_dir}/query_{query_id}_{method}.log"
    )
    logger.info(f"Start query {query_id} using method {method}")

    # Load data
    loader = MuSiQueDataLoader()
    query_info_dict, query_content_dict = loader.load_contents(query_ids=[query_id])
    query_info = query_info_dict[query_id]
    query_content = query_content_dict[query_id]
    chunks_by_page = {page: [text] for page, text in query_content.items()}

    # KG construction
    chunk_info_path = f"{output_dir}/query_{query_id}_kg.json"
    if os.path.exists(chunk_info_path):
        with open(chunk_info_path, "r") as f:
            chunk_info_list = json.load(f)
    else:
        constructor = NERConstructor(logger)
        chunk_info_list = []
        for page, chunks in chunks_by_page.items():
            chunk_info_list.extend(constructor.process_chunks(chunks, page))
        with open(chunk_info_path, "w") as f:
            json.dump(chunk_info_list, f, indent=2)

    chunks = [info["chunk"] for info in chunk_info_list]
    query = query_info["question"]
    correct_answers = [query_info["answer"]] + query_info["answer_aliases"]

    # QA method
    try:
        if method == "rawLLM":
            llm = LLM(verbose=True, logger=logger)
            response = llm.chat([{"role": "user", "content": query}])
        elif method == "RAG":
            rag = NaiveRAG(chunks, top_n=4, llm_verbose=True, logger=logger)
            response = rag.answer(query)
        elif method == "ToG":
            kg = KG([chunk_info_path])
            tog = ToG(
                kg,
                llm_verbose=True,
                logger=logger,
            )
            result = tog.answer(query, f"{output_dir}/query_{query_id}_{method}")
            response = result["prediction"]
        elif method == "FastToG":
            kg = KG([chunk_info_path])
            fasttog = FastToG(
                kg,
                llm_verbose=True,
                logger=logger,
            )
            result = fasttog.answer(query, f"{output_dir}/query_{query_id}_{method}")
            response = result["prediction"]
        else:
            raise ValueError(f"Unknown method: {method}")

        # Evaluate
        acc = loader.evaluate_answer(query, response, correct_answers, True, logger)
        logger.info(f"Final Accuracy: {acc:.2f}")
    except Exception as e:
        logger.error(f"Error during QA: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run KGQA Experiment with Specified Method"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["rawLLM", "RAG", "ToG", "FastToG"],
        help="QA method: rawLLM, RAG, ToG, FastToG",
    )
    parser.add_argument(
        "--query-ids",
        type=str,
        required=True,
        help="Comma-separated query IDs to run (e.g. 94,98,1041; available id range: 0 - 2416)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./exp_data",
        help="Directory to store outputs and logs",
    )

    args = parser.parse_args()
    method = args.method
    query_ids = [int(qid.strip()) for qid in args.query_ids.split(",")]
    query_ids = [qid for qid in query_ids if 0 <= qid <= 2416]
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for query_id in query_ids:
        run_single_query(query_id, method, output_dir)
