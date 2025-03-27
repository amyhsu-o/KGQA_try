from abc import ABC, abstractmethod
import json
import bz2
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import wikipedia
from bs4 import BeautifulSoup


class DataLoader(ABC):
    """
    >>> chunks = DataLoader.chunk(content)
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

    def load_contents(self, **kwargs) -> dict[int, dict[str, str]]:
        """query_ids: list[int], is required

        - key: query_id
        - value: dict of page_content

        >>> query_ids = [2, 54, 92, 98, 272]
        >>>
        >>> data_loader = CRAGDataLoader()
        >>> queries_content_dict = data_loader.load_contents(query_ids=query_ids)
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

        return queries_content_dict

    def _parse_html(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(" ", strip=True)


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

    # load query info
    queries_info_dict = data_loader.load_query_info(query_ids)
    for query_id, query_info in queries_info_dict.items():
        print(f"query_id: {query_id}")
        for key, info in query_info.items():
            print(
                f"{key}: {info if key != 'search_results' else f'{len(info)} ({info[0].keys()})'}"
            )
        print()

    # load content & chunk
    queries_content_dict = data_loader.load_contents(query_ids=query_ids)
    queries_chunks_dict = {}
    for query_id in query_ids:
        page_content_dict = queries_content_dict[query_id]
        page_chunks_dict = {
            page_name: data_loader.chunk(page_content)
            for page_name, page_content in page_content_dict.items()
        }
        queries_chunks_dict[query_id] = page_chunks_dict

    # show chunks info
    for query_id in query_ids:
        print(f"query_id: {query_id}")
        page_chunks_dict = queries_chunks_dict[query_id]
        for idx, (title, chunks) in enumerate(page_chunks_dict.items()):
            print(f"{title} - # of chunks: {len(chunks)}")
        print()
