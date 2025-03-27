import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import ast
from typing import Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from lm.ner import NER
from lm.llm import LLM


class NERConstructor:
    """
    >>> title = "Tom Hanks"
    >>>
    >>> # load content -> chunks
    >>> data_loader = WikiDataLoader()
    >>> titles_content_dict = data_loader.load_contents(titles=[title])
    >>> content = list(titles_content_dict.values())[0]
    >>> chunks = data_loader.chunk(content)
    >>> print(f"{title} - # of chunks: {len(chunks)}")
    >>>
    >>> # extract entities & relations
    >>> print(f"Start: {datetime.now()}")
    >>> kg_constructor = NERConstructor()
    >>> chunk_info_list = kg_constructor.process_chunks(chunks)
    >>> print(f"Done: {datetime.now()}")
    """

    def __init__(self):
        # models
        self.ner_model = NER()
        self.llm_model = LLM()

    def process_chunks(
        self, chunks: list[str], source: Optional[str] = None
    ) -> list[dict[str, any]]:
        """return
        - `source` (if given)
        - `chunk`
        - `entities`: dict[str, str]; Ex. {"Tom Hanks": "person"}
        - `triples`: list[dict[str, any]]; Ex. {"subject": "Tom Hanks", "relationship": "is", "object": "actor"}
        """
        chunk_info_list = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self._process_chunk, chunk, source) for chunk in chunks
            ]

            for future in tqdm(
                as_completed(futures), desc="Processing chunks", total=len(chunks)
            ):
                chunk_info = future.result()
                chunk_info_list.append(chunk_info)

        return chunk_info_list

    def _process_chunk(
        self, chunk: str, source: Optional[str] = None
    ) -> dict[str, any]:
        """return
        - `source` (if given)
        - `chunk`
        - `entities`: dict[str, str]; Ex. {"Tom Hanks": "person"}
        - `triples`: list[dict[str, any]]; Ex. {"subject": "Tom Hanks", "relationship": "is", "object": "actor"}
        """

        entities = self.extract_entities(chunk)
        triples = self.extract_triples(chunk, entities)

        chunk_info = {}
        if source:
            chunk_info["source"] = source
        chunk_info.update({"chunk": chunk, "entities": entities, "triples": triples})

        return chunk_info

    def extract_entities(self, chunk: str) -> dict[str, any]:
        """return entity format: 'thomas jeffrey hanks': 'person'"""
        return self.ner_model.extract_entities(chunk)

    def extract_triples(
        self, chunk: str, entities: dict[str, any]
    ) -> list[dict[str, any]]:
        """return relation format: {"subject": "John", "relationship": "lives in", "object": "US"}"""

        # ask LLM to extract triples
        system_message = """Extract all the relationships between the following entities ONLY based on the given ones.
        Return a list of JSON objects. For example:

        <Examples>
            [{{"subject": "John", "relationship": "lives in", "object": "US"}},
            {{"subject": "Eifel towel", "relationship": "is located in", "object": "Paris"}},
            {{"subject": "Hayao Miyazaki", "relationship": "is", "object": "Japanese animator"}}]
        </Examples>

        Note:
        1. ONLY return triples and nothing else. 
        2. None of "subject", "relationship" and "object" can be empty. 
        3. "Subject" and "object" should be a string and if it appears in entities list, please follow the text used in entities list.
        4. If many entities match, please write them separately into several triples.

        Entities: \n\n{entities}"""

        user_message = "Context: {text}\n\nTriples:"

        response = self.llm_model.chat(
            [
                {
                    "role": "system",
                    "content": system_message.format(entities=list(entities.keys())),
                },
                {
                    "role": "user",
                    "content": user_message.format(text=chunk),
                },
            ]
        )

        # extract triples from LLM response
        triples = self._parse_json_codeblock(
            response,
            r"""\{[\"']subject[\"']:\s*[\"'].*?[\"'],\s*[\"']relationship[\"']:\s*[\"'].*?[\"'],\s*[\"']object[\"']:\s*[\"'].*?[\"']\}""",
        )
        triples = [
            {key: value.lower() for key, value in triple.items()} for triple in triples
        ]

        return triples

    def _parse_json_codeblock(self, text: str, pattern: str) -> list[dict[str, any]]:
        matches = re.findall(
            pattern,
            text,
        )
        parsed_result = []
        for match in matches:
            try:
                parsed_result.append(ast.literal_eval(match))
            except SyntaxError:
                print(f"Failed to parse: {match}")
        return parsed_result


if __name__ == "__main__":
    import json
    from datetime import datetime
    from data_loader.data_loader import WikiDataLoader

    CHUNK_INFO_LIST_PATH = "chunk_info_list.json"

    if os.path.exists(CHUNK_INFO_LIST_PATH):
        print("Chunk info list already exists")
    else:
        title = "Tom Hanks"

        # load content -> chunks
        data_loader = WikiDataLoader()
        titles_content_dict = data_loader.load_contents(titles=[title])
        content = list(titles_content_dict.values())[0]
        chunks = data_loader.chunk(content)
        print(f"{title} - # of chunks: {len(chunks)}")

        # extract entities & relations
        print(f"Start: {datetime.now()}")
        kg_constructor = NERConstructor()
        chunk_info_list = kg_constructor.process_chunks(chunks, title)
        print(f"Done: {datetime.now()}")

        # count unique entities
        unique_entities = {}
        for chunk_info in chunk_info_list:
            entities = chunk_info["entities"]
            unique_entities.update(entities)
        print(f"# of entities: {len(unique_entities)}")

        # count triples
        print(
            f"# of triples: {sum([len(chunk_info['triples']) for chunk_info in chunk_info_list])}"
        )

        # save chunk info list
        with open(CHUNK_INFO_LIST_PATH, "w") as f:
            json.dump(
                chunk_info_list,
                f,
                indent=4,
            )
