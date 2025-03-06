import os
import json
import re
import logging
import argparse
from tqdm import tqdm
from gliner import GLiNER
from litellm import completion
import networkx as nx
from pyvis.network import Network
from prompt import (
    extract_relation_prompt,
    score_entity_candidates_prompt,
    prompt_evaluate,
    answer_prompt,
    cot_prompt,
)


def merge_entities(text, entities):
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


def extract_entities_for_query(
    chunks: list[str], labels: list[str]
) -> tuple[list[str], list[list[str]]]:
    """by NER model"""

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
            chunk_entities.add(entity["text"])
            if entity["text"] in duplicates:
                continue
            duplicates.add(entity["text"])
            entity_list.append((entity["text"], "=>", entity["label"]))

        chunks_entities.append(list(chunk_entities))

    return entity_list, chunks_entities


def run_llm(messages: list[dict[str, str]]) -> str:
    response = (
        completion(
            model="ollama/phi4",
            messages=messages,
            api_base="http://localhost:11435",
            max_tokens=1000,
        )
        .choices[0]
        .message.content
    )
    logging.info(messages[1]["content"])
    logging.info(response)
    return response


def generate_without_explored_paths(question):
    user_prompt = f"""Q: {question}\nA:"""
    response = run_llm(
        [
            {"content": cot_prompt, "role": "system"},
            {"content": user_prompt, "role": "user"},
        ]
    )
    return response


def extract_entities(path: str) -> tuple[list[str], list[list[str]]]:
    ENTITY_LIST_PATH = f"{path}/entities.txt"
    CHUNKS_ENTITIES_PATH = f"{path}/chunks_entities.txt"
    if os.path.exists(ENTITY_LIST_PATH) and os.path.exists(CHUNKS_ENTITIES_PATH):
        logging.info("load entities from file")

        with open(ENTITY_LIST_PATH, "r") as f:
            entity_list = json.load(f)
        with open(CHUNKS_ENTITIES_PATH, "r") as f:
            chunks_entities = json.load(f)
        return entity_list, chunks_entities
    else:
        logging.info("entities.txt & chunks_entities.txt not found")
        return [], []


def classify_entities(entity_list: list[str], labels: list[str]) -> dict[list[str]]:
    labels_entities = {label: set() for label in labels}

    for e in entity_list:
        s, _, o = e
        labels_entities[o].add(s)

    return labels_entities


def extract_triples(path: str) -> tuple[list[list[str]], list[str]]:
    """given entities output triples -- by LLM"""
    TRIPLES_PATH = f"{path}/chunks_triples.txt"
    ERRORS_PATH = f"{path}/errors.txt"
    if os.path.exists(TRIPLES_PATH) and os.path.exists(ERRORS_PATH):
        logging.info("load triples from file")

        with open(TRIPLES_PATH, "r") as f:
            chunks_triples = json.load(f)
        with open(ERRORS_PATH, "r") as f:
            errors = json.load(f)
        return chunks_triples, errors
    else:
        logging.info("chunks_triples.txt & errors.txt not found")
        return [], []


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
    chunks_triples: list[list[str]], labels_entities: dict[str, list[str]]
) -> nx.Graph:
    G = nx.Graph()

    for items in chunks_triples:
        for item in items:
            try:
                node1 = item["subject"].lower()
                node2 = item["object"].lower()
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
            except Exception as e:
                logging.error(f"{e} in item: {item}")

    return G


def show_adjacent_relations(all_relations: list[dict[str]]) -> None:
    for relation in all_relations:
        logging.info(f"{relation['head']} -- {relation['title']} -- {relation['tail']}")


def is_unused_triple(
    description: dict[str, any], used_triples: dict[str, list[tuple[any]]]
) -> bool:
    if description["head"] not in used_triples:
        return True

    for triple in used_triples[description["head"]]:
        if triple[1] == description["title"] and triple[2] == description["tail"]:
            return False

    return True


def unused_relation_search(
    G: nx.Graph, entity: str, cluster_chain_of_entities: list[list[tuple[any]]]
) -> list[dict[str, any]]:
    used_triples = {}
    for cluster in cluster_chain_of_entities:
        for triple in cluster:
            head = triple[0]
            used_triples[head] = used_triples.get(head, [])
            used_triples[head].append(triple)

    unused_relations = [
        description
        for _, description in G[entity].items()
        if is_unused_triple(description, used_triples)
    ]
    return unused_relations


def clean_relations(response: str, entity: str) -> tuple[bool, list[str]]:
    patterns = [
        r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}",
        r"\*\*\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)\*\*",
    ]
    relations = []
    for pattern in patterns:
        for match in re.finditer(pattern, response):
            try:
                relation = match.group("relation").strip()
                score = float(match.group("score"))
                relations.append(
                    {
                        "entity": entity,
                        "relation": relation.lower(),
                        "score": score,
                    }
                )
            except ValueError:
                logging.error(f"Error in item: {match}")

        if len(relations) > 0:
            break

    return relations


def relation_prune(
    entity: str,
    total_relations: list[dict[str]],
    question: str,
):
    total_relations.sort(key=lambda x: x["title"])
    total_relations = list(set([relation["title"] for relation in total_relations]))

    # ask LLM to choose top k relations
    system_prompt = extract_relation_prompt.format(top_k=TOP_K)
    user_prompt = f"""Q: {question}
Topic Entity: {entity}
Relations: {"; ".join(total_relations)}
A: """
    response = run_llm(
        [
            {"content": system_prompt, "role": "system"},
            {"content": user_prompt, "role": "user"},
        ]
    )

    # clean the response
    retrieve_relations_with_scores = clean_relations(response, entity)

    if len(retrieve_relations_with_scores) == 0:
        return []
    return retrieve_relations_with_scores


def entity_search(G: nx.Graph, entity: str, relation: str) -> list[str]:
    adjacent_edges_with_relation = {}

    for _, description in G[entity].items():
        if description["title"] == relation:
            if description["head"] == entity:
                adjacent_entity = description["tail"]
            elif description["tail"] == entity:
                adjacent_entity = description["head"]
            adjacent_edges_with_relation[adjacent_entity] = description

    return adjacent_edges_with_relation


def clean_scores(response: str, entity_candidates: list[str]) -> list[float]:
    scores = re.findall(r"\d+\.\d+", response)
    scores = [float(score) for score in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        return [1 / len(entity_candidates)] * len(entity_candidates)


def entity_score(
    question: str, relation: str, entity_candidates: list[str], score: float
):
    user_prompt = f"""Q: {question}
Relation: {relation}
Entities: {entity_candidates}"""
    response = run_llm(
        [
            {"content": score_entity_candidates_prompt, "role": "system"},
            {"content": user_prompt, "role": "user"},
        ]
    )
    scores = [float(x) * score for x in clean_scores(response, entity_candidates)]
    return scores


def entity_prune(total_selected_edges: list[dict[str, any]]) -> list[dict[str, any]]:
    total_selected_edges.sort(key=lambda x: x["score"], reverse=True)
    return total_selected_edges[:TOP_K]


def get_new_topic_entities(
    important_edges: list[dict[str, any]], topic_entities: list[str]
) -> list[str]:
    new_topic_entities = set()
    for edge in important_edges:
        if edge["head"] not in topic_entities:
            new_topic_entities.add(edge["head"])
        if edge["tail"] not in topic_entities:
            new_topic_entities.add(edge["tail"])
    return new_topic_entities


def reasoning(question: str, cluster_chain_of_entities: list[list[tuple[any]]]) -> str:
    chain_prompt = "\n".join(
        [
            ", ".join(str(x) for x in chain)
            for sublist in cluster_chain_of_entities
            for chain in sublist
        ]
    )
    user_prompt = f"""Q: {question}
Knowledge Triplets: {chain_prompt}
A: """

    response = run_llm(
        [
            {"content": prompt_evaluate, "role": "system"},
            {"content": user_prompt, "role": "user"},
        ]
    )
    return response


def extract_answer(text: str) -> str:
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        answer = text[start_index + 1 : end_index].strip()
        if answer.lower().strip().replace(" ", "") == "yes":
            return True
    return False


def save_current_cluster_chain(
    path: str, cluster_chain_of_entities: list[list[tuple[any]]], round_count: int
) -> None:
    all_triples = [[]]
    for cluster in cluster_chain_of_entities:
        for triple in cluster:
            all_triples[0].append(
                {"subject": triple[0], "relationship": triple[1], "object": triple[2]}
            )

    tempG = draw_graph(all_triples, labels_entities)
    nt = Network(height="750px", width="100%")
    nt.from_nx(tempG)
    nt.force_atlas_2based(central_gravity=0.015, gravity=-31)
    nt.save_graph(f"{path}/reasoning_path/graph{round_count}.html")


def generate_answer(
    question: str, cluster_chain_of_entities: list[list[tuple[any]]]
) -> tuple[str, str]:
    chain_prompt = "\n".join(
        [
            ", ".join(str(x) for x in chain)
            for sublist in cluster_chain_of_entities
            for chain in sublist
        ]
    )
    user_prompt = f"""Q: {question}
Knowledge Triplets: {chain_prompt}
A: """
    response = run_llm(
        [
            {"content": answer_prompt, "role": "system"},
            {"content": user_prompt, "role": "user"},
        ]
    )
    return user_prompt, response


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

TOP_K = 10

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", type=str, required=True)
    arg_parser.add_argument("--query", type=str, required=True)
    args = arg_parser.parse_args()

    # path
    DATA_PATH = args.path
    if not os.path.exists(f"{DATA_PATH}/construction"):
        exit()
    if not os.path.exists(f"{DATA_PATH}/reasoning_path"):
        os.makedirs(f"{DATA_PATH}/reasoning_path")

    # logging
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"{DATA_PATH}/log/reasoning_path.log",
                mode="w",
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )

    # === Query ===
    question = args.query
    _, chunks_entities = extract_entities_for_query([question], LABELS)
    topic_entities = [entity.lower() for entity in chunks_entities[0]]

    # === LLM only ===
    response_without_explored_paths = generate_without_explored_paths(question)

    # === Create Graph ===
    entity_list, chunks_entities = extract_entities(f"{DATA_PATH}/construction")
    logging.info(f"# of entities: {len(entity_list)}")

    labels_entities = classify_entities(entity_list, LABELS)
    logging.info({label: len(entities) for label, entities in labels_entities.items()})

    chunks_triples, errors = extract_triples(f"{DATA_PATH}/construction")
    logging.info(f"# of triples: {sum(len(triples) for triples in chunks_triples)}")

    G = draw_graph(chunks_triples, labels_entities)

    # === ToG ===
    done = False
    round_count = 1
    cluster_chain_of_entities = []

    while not done:
        logging.info(f"\n===== Round {round_count} =====")

        # === relation ===
        current_entity_relations_list = []

        for idx, entity in enumerate(topic_entities):
            try:
                all_relations = unused_relation_search(
                    G, entity, cluster_chain_of_entities
                )
                if len(all_relations) == 0:
                    logging.info(f"{entity}: 0 relation selected")
                    continue

                retrieve_relations_with_scores = relation_prune(
                    entity,
                    all_relations,
                    question,
                )
                current_entity_relations_list.extend(retrieve_relations_with_scores)

                logging.info(
                    f"{entity}: {len(retrieve_relations_with_scores)} relations selected"
                )
            except KeyError:
                logging.error(f"{entity}: not found")

        logging.info(
            f"\ncurrent_entity_relations_list:\n{current_entity_relations_list}"
        )

        if len(current_entity_relations_list) == 0:
            break

        # === entity ===
        total_selected_edges = []

        for entity_relation in current_entity_relations_list:
            entity = entity_relation["entity"]
            relation = entity_relation["relation"]
            score = entity_relation["score"]

            # find all adjacent nodes with the relation
            adjacent_edges_with_relation = entity_search(G, entity, relation)
            entity_candidates = list(adjacent_edges_with_relation.keys())

            if len(entity_candidates) == 0:
                continue

            # ask LLM to score the entities
            scores = entity_score(
                question,
                entity_relation["relation"],
                entity_candidates,
                entity_relation["score"],
            )

            # add score to edges
            for idx, entity in enumerate(entity_candidates):
                adjacent_edges_with_relation[entity]["score"] = scores[idx]

            total_selected_edges.extend(list(adjacent_edges_with_relation.values()))

        logging.info(f"\ntotal_selected_edges:\n{total_selected_edges}")

        important_edges = entity_prune(total_selected_edges)
        logging.info(f"\nimportant_edges:\n{important_edges}")

        # update
        topic_entities = get_new_topic_entities(important_edges, topic_entities)

        if len(important_edges) > 0:
            logging.info("\nSelected edges:")
            show_adjacent_relations(important_edges)

            important_edges = [
                (edge["head"], edge["title"], edge["tail"]) for edge in important_edges
            ]
            cluster_chain_of_entities.append(important_edges)

            save_current_cluster_chain(
                DATA_PATH, cluster_chain_of_entities, round_count
            )

            response = reasoning(question, cluster_chain_of_entities)
            done = extract_answer(response)
        else:
            logging.info("No edges selected")
            done = True

        round_count += 1

    # === Answer ===
    prompt, response = generate_answer(question, cluster_chain_of_entities)
    logging.info(f"\nprompt:\n{prompt}")
    logging.info(f"\nresponse:\n{response}")
