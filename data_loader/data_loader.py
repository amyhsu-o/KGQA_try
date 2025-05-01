from abc import ABC, abstractmethod
import json
import bz2
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import wikipedia
from bs4 import BeautifulSoup
from lm import LLM


class DataLoader(ABC):
    """
    >>> chunks = DataLoader.chunk(content)
    """

    evaluation_prompt = """
# Task: 
You are given a Question, a model Prediction, and a list of Ground Truth answers, judge whether the model Prediction matches any answer from the list of Ground Truth answers. Follow the instructions step by step to make a judgement. 
1. If the model prediction matches any provided answers from the Ground Truth Answer list, "Accuracy" should be "True"; otherwise, "Accuracy" should be "False".
2. If the model prediction says that it couldn't answer the question or it doesn't have enough information, "Accuracy" should always be "False".
3. If the Ground Truth is "invalid question", "Accuracy" is "True" only if the model prediction is exactly "invalid question".
# Output: 
Respond with only a single JSON string with an "Accuracy" field which is "True" or "False".

# Examples:
Question: how many seconds is 3 minutes 15 seconds?
Ground truth: ["195 seconds"]
Prediction: 3 minutes 15 seconds is 195 seconds.
Accuracy: True

Question: Who authored The Taming of the Shrew (published in 2002)?
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: The author to The Taming of the Shrew is Roma Shakespeare.
Accuracy: False

Question: Who played Sheldon in Big Bang Theory?
Ground truth: ["Jim Parsons", "Iain Armitage"]
Prediction: I am sorry I don't know.
Accuracy: False
"""

    @abstractmethod
    def load_contents(self, **kwargs):
        pass

    @staticmethod
    def chunk(content: str) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=50, separators=["\n\n", "\n", ". "]
        )
        chunks = text_splitter.split_text(content)
        return chunks

    @staticmethod
    def get_source_label(index: int, title: str) -> str:
        return f"({index}) {title}"

    @staticmethod
    def evaluate_answer(
        query: str,
        prediction: str,
        correct_answers: list[str],
        llm_verbose=False,
        logger=None,
    ) -> dict[str, any]:
        """evaluate answer with LLM"""
        llm = LLM(verbose=llm_verbose, logger=logger)
        user_prompt = f"""Question: {query}
Ground truth: {correct_answers}

Prediction:
{prediction}"""
        response = llm.chat(
            [
                {"role": "system", "content": DataLoader.evaluation_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        )
        response = llm.parse_response(response, r"(\{\n.*?\n\})", {"answer": str})
        if len(response) == 0:
            return False
        if "true" in response[0]["answer"].lower():
            return True
        else:
            return False


class WikiDataLoader(DataLoader):
    def load_contents(self, **kwargs) -> dict[str, str]:
        """wikipedia title names (titles=): list[str], are required

        >>> data_loader = WikiDataLoader()
        >>> titles_content_dict = data_loader.load_contents(titles=["Tom Hanks", "Brad Pitt"])
        """
        if "titles" not in kwargs:
            raise ValueError("Need to input title name")

        titles = kwargs["titles"]
        titles_content_dict = {}
        for idx, title in enumerate(titles):
            content = wikipedia.page(title=title, auto_suggest=False).content
            titles_content_dict[self.get_source_label(idx, title)] = content
        return titles_content_dict


class CRAGDataLoader(DataLoader):
    def load_query_info(self, query_ids: list[int]) -> dict[int, dict[str, any]]:
        """output format
        - key: query_id
        - value: query_info
            - key: interaction_id, query_time, domain, question_type, static_or_dynamic, query, answer, search_results, split, alt_ans
            - search_result: dict_keys(['page_name', 'page_url', 'page_snippet', 'page_result', 'page_last_modified']

        >>> query_ids = [2, 54, 92, 98, 272]
        >>>
        >>> data_loader = CRAGDataLoader()
        >>> queries_info_dict = data_loader.load_query_info(query_ids)
        """
        CRAG_DATA_PATH = "./data/crag_task_1_dev_v4_release.jsonl.bz2"

        queries_info_dict = {}
        with bz2.open(CRAG_DATA_PATH, "rt") as f:
            for idx, line in tqdm(
                enumerate(f), desc="Finding query info", total=max(query_ids)
            ):
                if idx in query_ids:
                    data = json.loads(line)
                    queries_info_dict[idx] = data
                if idx == max(query_ids):
                    break
        return queries_info_dict

    def load_contents(
        self, **kwargs
    ) -> tuple[dict[int, dict[str, any]], dict[int, dict[str, str]]]:
        """query_ids: list[int], is required

        queries_info_dict
        - key: query_id
        - value: query_info
            - key: interaction_id, query_time, domain, question_type, static_or_dynamic, query, answer, search_results, split, alt_ans
            - search_result: dict_keys(['page_name', 'page_url', 'page_snippet', 'page_result', 'page_last_modified']

        queries_content_dict
        - key: query_id
        - value: dict of page_content

        >>> query_ids = [2, 54, 92, 98, 272]
        >>>
        >>> data_loader = CRAGDataLoader()
        >>> queries_info_dict, queries_content_dict = data_loader.load_contents(query_ids=query_ids)
        """
        if "query_ids" not in kwargs:
            raise ValueError("Need to input query_ids")

        query_ids = kwargs["query_ids"]
        queries_info_dict = self.load_query_info(query_ids)

        queries_content_dict = {}
        for query_id in query_ids:
            query_info = queries_info_dict[query_id]

            # get raw contents for all search results
            raw_contents = {
                search_result["page_result"]: search_result["page_name"]
                for search_result in query_info["search_results"]
                if len(search_result["page_result"]) > 0
            }

            # parse html to text
            contents = {
                self.get_source_label(idx, page_name): self._parse_html(raw_content)
                for idx, (raw_content, page_name) in enumerate(raw_contents.items())
            }

            queries_content_dict[query_id] = contents

        return queries_info_dict, queries_content_dict

    def _parse_html(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(" ", strip=True)


class MuSiQueDataLoader(DataLoader):
    @staticmethod
    def get_decomposition_type(decomposition_process: list[dict[str, any]]) -> int:
        """decomposition_process: from question_decomposition"""
        if len(decomposition_process) == 2:
            return 0
        if len(decomposition_process) == 3:
            if "#1" in decomposition_process[2]["question"]:
                return 2
            else:
                return 1
        if len(decomposition_process) == 4:
            if "#2" in decomposition_process[3]["question"]:
                return 5
            elif "#1" in decomposition_process[2]["question"]:
                return 4
            else:
                return 3

    def load_query_info(self, query_ids: list[int]) -> dict[int, dict[str, any]]:
        """output format
        - key: query_id
        - value: query_info
            - key: 'id', 'paragraphs', 'question', 'question_decomposition', 'answer', 'answer_aliases', 'answerable', 'decomposition_type'
        """
        MUSIQUE_DATA_PATH = "./data/musique_ans_v1.0_dev.jsonl"

        queries_info_dict = {}
        with open(MUSIQUE_DATA_PATH) as f:
            for idx, line in enumerate(f):
                if idx in query_ids:
                    data = json.loads(line)
                    data["decomposition_type"] = self.get_decomposition_type(
                        data["question_decomposition"]
                    )
                    queries_info_dict[idx] = data
                if idx == max(query_ids):
                    break
        return queries_info_dict

    def load_contents(
        self, **kwargs
    ) -> tuple[dict[int, dict[str, any]], dict[int, dict[str, str]]]:
        """query_ids: list[int], is required

        queries_info_dict
        - key: query_id
        - value: query_info
            - key: 'id', 'paragraphs', 'question', 'question_decomposition', 'answer', 'answer_aliases', 'answerable', 'decomposition_type'

        queries_content_dict
        - key: query_id
        - value: dict of page_content
        """

        if "query_ids" not in kwargs:
            raise ValueError("Need to input query_ids")

        query_ids = kwargs["query_ids"]
        queries_info_dict = self.load_query_info(query_ids)

        queries_content_dict = {}
        for query_id in query_ids:
            query_info = queries_info_dict[query_id]
            contents = {
                self.get_source_label(paragraph["idx"], paragraph["title"]): paragraph[
                    "paragraph_text"
                ]
                for paragraph in query_info["paragraphs"]
            }
            queries_content_dict[query_id] = contents

        return queries_info_dict, queries_content_dict


if __name__ == "__main__":
    # === Wiki ===
    data_loader = WikiDataLoader()
    titles_content_dict = data_loader.load_contents(titles=["Tom Hanks", "Brad Pitt"])
    titles_chunks_dict = {
        title: data_loader.chunk(content)
        for title, content in titles_content_dict.items()
    }
    for title, chunks in titles_chunks_dict.items():
        print(f"{title} - # of chunks: {len(chunks)}")
    print()

    # === CRAG ===
    query_ids = [2, 54, 92, 98, 272]
    data_loader = CRAGDataLoader()

    queries_info_dict, queries_content_dict = data_loader.load_contents(
        query_ids=query_ids
    )

    # chunk contents
    queries_chunks_dict = {}
    for query_id in query_ids:
        page_content_dict = queries_content_dict[query_id]
        page_chunks_dict = {
            page_name: data_loader.chunk(page_content)
            for page_name, page_content in page_content_dict.items()
        }
        queries_chunks_dict[query_id] = page_chunks_dict

    # show query info
    for query_id in query_ids:
        print(f"query_id: {query_id}")
        # query info
        for key, info in queries_info_dict[query_id].items():
            print(
                f"{key}: {info if key != 'search_results' else f'{len(info)} ({info[0].keys()})'}"
            )

        # chunk info
        page_chunks_dict = queries_chunks_dict[query_id]
        for idx, (title, chunks) in enumerate(page_chunks_dict.items()):
            print(f"{title} - # of chunks: {len(chunks)}")
        print()
