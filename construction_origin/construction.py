import os
import re
import json
from json import JSONDecodeError
from tqdm import tqdm
import logging
import argparse
import wikipedia
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gliner import GLiNER
from litellm import completion
import networkx as nx
from pyvis.network import Network


def create_dir(title: str) -> None:
    TITLE_DIR = f"./{''.join(title.split(' '))}"
    if not os.path.exists(TITLE_DIR):
        os.makedirs(TITLE_DIR)


def load_wikipedia_page(title: str) -> str:
    PAGE_CONTENT_PATH = f"./{''.join(title.split(' '))}/construction/page.txt"
    if os.path.exists(PAGE_CONTENT_PATH):
        logging.info("load page content from file")

        with open(PAGE_CONTENT_PATH, "r") as f:
            page_content = f.read()
    else:
        logging.info("load page content from wikipedia")

        page_content = wikipedia.page(title=title, auto_suggest=False).content

        with open(PAGE_CONTENT_PATH, "w") as f:
            f.write(page_content)
    return page_content


def chunk_page_content(title: str, page_content: str) -> list[str]:
    CHUNKS_PATH = f"./{''.join(title.split(' '))}/construction/chunks.txt"
    if os.path.exists(CHUNKS_PATH):
        logging.info("load chunks from file")

        with open(CHUNKS_PATH, "r") as f:
            chunks = json.load(f)
    else:
        logging.info("chunk the page content")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=20, separators=["\n\n", "\n"]
        )
        chunks = text_splitter.split_text(page_content)

        with open(CHUNKS_PATH, "w") as f:
            json.dump(chunks, f, indent=4)
    return chunks


def merge_entities(text: str, entities: list[dict[str, any]]) -> list[dict[str, any]]:
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity["label"] == current["label"] and (
            next_entity["start"] == current["end"] + 1
            or next_entity["start"] == current["end"]
        ):
            current["text"] = text[current["start"] : next_entity["end"]].strip()
            current["end"] = next_entity["end"]
        else:
            merged.append(current)
            current = next_entity

    # Append the last entity
    merged.append(current)
    return merged


def extract_entities(
    title: str, chunks: list[str], labels: list[str]
) -> tuple[list[str], list[list[str]]]:
    """by NER model"""
    ENTITY_LIST_PATH = f"./{''.join(title.split(' '))}/construction/entities.txt"
    CHUNKS_ENTITIES_PATH = (
        f"./{''.join(title.split(' '))}/construction/chunks_entities.txt"
    )
    if os.path.exists(ENTITY_LIST_PATH) and os.path.exists(CHUNKS_ENTITIES_PATH):
        logging.info("load entities from file")

        with open(ENTITY_LIST_PATH, "r") as f:
            entity_list = json.load(f)
        with open(CHUNKS_ENTITIES_PATH, "r") as f:
            chunks_entities = json.load(f)
    else:
        logging.info("extract entities from chunks")

        model = GLiNER.from_pretrained("numind/NuNerZero")
        entity_list = []
        chunks_entities = []
        duplicates = set()

        for text in tqdm(chunks):
            entities = model.predict_entities(text, labels, threshold=0.7)
            entities = merge_entities(text, entities)
            chunk_entities = set()

            for entity in entities:
                entity["text"] = entity["text"].lower()
                chunk_entities.add(entity["text"])
                if entity["text"] in duplicates:
                    continue
                duplicates.add(entity["text"])
                entity_list.append((entity["text"], "=>", entity["label"]))

            chunks_entities.append(list(chunk_entities))

        with open(ENTITY_LIST_PATH, "w") as f:
            json.dump(entity_list, f, indent=4)
        with open(CHUNKS_ENTITIES_PATH, "w") as f:
            json.dump(chunks_entities, f, indent=4)
    return entity_list, chunks_entities


def classify_entities(entity_list: list[str], labels: list[str]) -> dict[list[str]]:
    labels_entities = {label: set() for label in labels}

    for e in entity_list:
        s, _, o = e
        labels_entities[o].add(s)

    return labels_entities


def extract_json_list(response: str) -> list[str]:
    json_str_list = re.findall("{.+}", response)
    return [json.loads(json_str) for json_str in json_str_list]


def extract_triples(
    title: str, chunks: list[str], chunks_entities: list[list[str]]
) -> tuple[list[list[str]], list[str]]:
    """given entities output triples -- by LLM"""
    TRIPLES_PATH = f"./{''.join(title.split(' '))}/construction/chunks_triples.txt"
    ERRORS_PATH = f"./{''.join(title.split(' '))}/construction/errors.txt"
    if os.path.exists(TRIPLES_PATH) and os.path.exists(ERRORS_PATH):
        logging.info("load triples from file")

        with open(TRIPLES_PATH, "r") as f:
            chunks_triples = json.load(f)
        with open(ERRORS_PATH, "r") as f:
            errors = json.load(f)
    else:
        logging.info("extract triples from chunks")

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

        errors = []
        chunks_triples = []

        for i in tqdm(range(len(chunks_entities))):
            try:
                text = chunks[i]
                ents = "\n\n".join(chunks_entities[i])

                response = (
                    completion(
                        model="ollama/phi4",
                        messages=[
                            {
                                "content": system_message.format(entities=ents),
                                "role": "system",
                            },
                            {"content": user_message.format(text=text), "role": "user"},
                        ],
                        api_base="http://localhost:11435",
                        max_tokens=1000,
                    )
                    .choices[0]
                    .message.content
                )

                triples = extract_json_list(response)
                additional_triples = []
                for idx, triple in enumerate(triples):
                    for key in triple:
                        print(triples[idx])
                        if triples[idx][key] is None:
                            continue
                        elif isinstance(triples[idx][key], list):
                            for item in triples[idx][key]:
                                additional_triples.append(
                                    {
                                        "subject": triple["subject"],
                                        "relationship": triple["relationship"],
                                        "object": item.lower(),
                                    }
                                )
                        else:
                            triples[idx][key] = triples[idx][key].lower()
                chunks_triples.append(triples + additional_triples)

                logging.info(f"Chunks: {text}")
                logging.info(f"Entities: {'; '.join(chunks_entities[i])}")
                logging.info(f"Triples: {triples}")
            except JSONDecodeError as e:
                errors.append(response)
                chunks_triples.append([])

                logging.info(f"Chunks: {text}")
                logging.info(f"{e} in chunk {i}")

        with open(TRIPLES_PATH, "w") as f:
            json.dump(chunks_triples, f, indent=4)
        with open(ERRORS_PATH, "w") as f:
            json.dump(errors, f, indent=4)

    return chunks_triples, errors


def get_color(label: str, labels_entities: dict[str, list[str]]) -> str:
    colors = [
        "orange",
        "blue",
        "green",
        "brown",
        "red",
        "purple",
        "yellow",
        "pink",
        "cyan",
        "magenta",
        "lime",
        "teal",
        "navy",
        "gold",
        "silver",
        "violet",
        "coral",
        "indigo",
        "salmon",
        "turquoise",
        "black",
    ]
    for idx, label_entities in enumerate(list(labels_entities.values())):
        if label in label_entities:
            return colors[idx]
    return colors[-1]


def get_size(label: str, labels_entities: dict[str, list[str]]) -> int:
    sizes = list(range(40, 0, -2))
    for idx, label_entities in enumerate(list(labels_entities.values())):
        if label in label_entities:
            return sizes[idx]
    return sizes[-1]


def draw_graph(
    title: str, chunks_triples: list[list[str]], labels_entities: dict[str, list[str]]
) -> None:
    G = nx.Graph()

    for items in chunks_triples:
        for item in items:
            try:
                node1 = item["subject"]
                node2 = item["object"]
                G.add_node(
                    node1,
                    title=str(node1),
                    color=get_color(node1, labels_entities),
                    size=get_size(node1, labels_entities),
                    label=str(node1),
                )
                G.add_node(
                    node2,
                    title=str(node2),
                    color=get_color(node2, labels_entities),
                    size=get_size(node2, labels_entities),
                    label=str(node2),
                )
                G.add_edge(
                    node1,
                    node2,
                    title=str(item["relationship"]),
                    weight=4,
                    head=str(node1),
                    tail=str(node2),
                )
            except Exception:
                logging.info(f"Error in item: {item}")

    nt = Network(height="750px", width="100%")
    nt.from_nx(G)
    nt.force_atlas_2based(central_gravity=0.015, gravity=-31)
    nt.save_graph(f"./{''.join(title.split(' '))}/construction/graph.html")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--title", type=str, required=True)
    args = arg_parser.parse_args()

    TITLE = args.title
    LABELS = [
        "person",
        "organization",
        "location",
        "event",
        "date",
        "product",
        "law",
        "medical",
        "scientific_term",
        "work_of_art",
        "language",
        "nationality",
        "religion",
        "sport",
        "weapon",
        "food",
        "currency",
        "disease",
        "animal",
        "plant",
    ]

    create_dir(TITLE)
    create_dir(f"{TITLE}/construction")
    create_dir(f"{TITLE}/log")

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"./{''.join(TITLE.split(' '))}/log/construction.log",
                mode="w",
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logging.info(f"Start construction for {TITLE}")

    page_content = load_wikipedia_page(TITLE)

    chunks = chunk_page_content(TITLE, page_content)
    logging.info(f"# of chunks: {len(chunks)}")

    entity_list, chunks_entities = extract_entities(TITLE, chunks, LABELS)
    logging.info(f"# of entities: {len(entity_list)}")

    labels_entities = classify_entities(entity_list, LABELS)
    logging.info({label: len(entities) for label, entities in labels_entities.items()})

    chunks_triples, errors = extract_triples(TITLE, chunks, chunks_entities)
    logging.info(f"# of triples: {sum(len(triples) for triples in chunks_triples)}")

    draw_graph(TITLE, chunks_triples, labels_entities)
