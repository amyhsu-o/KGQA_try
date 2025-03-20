import os
import json
import re
import logging
import argparse
import numpy as np
import pandas as pd
import pickle
from litellm import completion
from ollama import Client
import networkx as nx
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity
from qa.ToG.prompt import (
    extract_relation_prompt,
    score_entity_candidates_prompt,
    extract_triple_prompt,
    prompt_evaluate,
    answer_prompt,
    cot_prompt,
)
from construction.construction import (
    extract_entities,
    classify_entities,
    extract_triples,
    draw_graph,
)


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


def get_text_embeddings(text_list: list[str]) -> list[np.ndarray]:
    embeddings = (
        Client(host="http://localhost:11435")
        .embed(model="nomic-embed-text:latest", input=text_list)
        .embeddings
    )
    embeddings = [np.array(embedding) for embedding in embeddings]
    return embeddings


def get_all_entity_embeddings(G: nx.Graph, path: str) -> dict[str, np.ndarray]:
    if os.path.exists(path):
        logging.info("load entity embeddings from file")
        with open(path, "rb") as f:
            entity_embeddings = pickle.load(f)
    else:
        logging.info("embed entities")
        entity_embeddings = {
            entity: embedding
            for entity, embedding in zip(
                list(G.nodes), get_text_embeddings(list(G.nodes))
            )
        }
        with open(path, "wb") as f:
            pickle.dump(entity_embeddings, f)
    return entity_embeddings


def retrieve_top_k_similarity(
    source_embeddings_df: pd.DataFrame, target_embedding: np.ndarray, top_k: int
) -> list[str]:
    temp_score_df = pd.DataFrame(
        data=cosine_similarity(source_embeddings_df, target_embedding.reshape(1, -1)),
        columns=["cosine_similarity"],
        index=source_embeddings_df.index,
    )
    temp_score_df = temp_score_df.sort_values(
        by="cosine_similarity", ascending=False
    ).head(top_k)
    return list(temp_score_df.index)


def generate_without_explored_paths(question):
    user_prompt = f"""Q: {question}\nA:"""
    response = run_llm(
        [
            {"content": cot_prompt, "role": "system"},
            {"content": user_prompt, "role": "user"},
        ]
    )
    return response


def show_adjacent_relations(all_relations: list[dict[str]]) -> None:
    for relation in all_relations:
        logging.info(f"{relation['head']} -- {relation['title']} -- {relation['tail']}")


def is_unused_triple(
    description: dict[str, any], used_triples: dict[str, list[tuple[any]]]
) -> bool:
    if description["head"] not in used_triples:
        return True

    for triple in used_triples[description["head"]]:
        if (
            triple["title"] == description["title"]
            and triple["tail"] == description["tail"]
        ):
            return False

    return True


def unused_adjacent_triple_search(
    G: nx.Graph, entity: str, cluster_chain_of_entities: list[list[dict[str, any]]]
) -> list[dict[str, any]]:
    """
    Return format:

    Ex. {'title': 'is a subsidiary of', 'weight': 4, 'head': 'salesforce.com', 'tail': 'salesforce'}
    """
    used_triples = {}
    for cluster in cluster_chain_of_entities:
        for triple in cluster:
            head = triple["head"]
            used_triples[head] = used_triples.get(head, [])
            used_triples[head].append(triple)

    unused_relations = [
        description
        for _, description in G[entity].items()
        if is_unused_triple(description, used_triples)
    ]
    return unused_relations


def clean_relations(response: str, entity: str) -> list[dict[str, any]]:
    patterns = [
        r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}",
        r"\*\*\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\):?\*\*",
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
    all_relations: list[dict[str, any]],
    question: str,
    method_to_score: str,
) -> list[dict[str, any]]:
    """
    Return format:

    Ex. {'entity': 'salesforce', 'relation': 'co-founded', 'score': 0.4}
    """
    all_relations.sort(key=lambda x: x["title"])
    all_relations = list(set([relation["title"] for relation in all_relations]))

    if method_to_score in ["llm", "llm_embedding"]:
        if len(all_relations) == 1:
            retrieve_relations_with_scores = [
                {
                    "entity": entity,
                    "relation": all_relations[0].lower(),
                    "score": 1,
                }
            ]
        else:
            # ask LLM to choose top k relations
            system_prompt = extract_relation_prompt.format(top_k=TOP_K)
            user_prompt = f"""Q: {question}
Topic Entity: {entity}
Relations: {"; ".join(all_relations)}
A: """
            response = run_llm(
                [
                    {"content": system_prompt, "role": "system"},
                    {"content": user_prompt, "role": "user"},
                ]
            )

            # clean the response
            retrieve_relations_with_scores = clean_relations(response, entity)
            logging.info(
                f"relations retrieved by LLM: {[relation['relation'] for relation in retrieve_relations_with_scores]}"
            )

        # check relations retrieved by LLM are in graph
        if method_to_score == "llm":
            substitute_relations_with_scores = []
            for relation in retrieve_relations_with_scores:
                if relation["relation"] in all_relations:
                    logging.info(f"Relation '{relation['relation']}' found")
                    substitute_relations_with_scores.append(relation)
                else:
                    logging.error(f"Relation '{relation['relation']}' not found")
            retrieve_relations_with_scores = substitute_relations_with_scores
        elif method_to_score == "llm_embedding":
            # relation embedding
            relation_embeddings = {
                relation: embedding
                for relation, embedding in zip(
                    all_relations, get_text_embeddings(all_relations)
                )
            }
            relation_embeddings_df = pd.DataFrame(relation_embeddings).T

            substitute_relations_with_scores = {}
            for relation in retrieve_relations_with_scores:
                # if relation not found, substitute with similar relation
                if relation["relation"] not in all_relations:
                    substitute_relation = retrieve_top_k_similarity(
                        relation_embeddings_df,
                        get_text_embeddings([relation["relation"]])[0],
                        1,
                    )[0]
                    logging.info(
                        f"Relation '{relation['relation']}' not found -> change to '{substitute_relation}'"
                    )
                else:
                    substitute_relation = relation["relation"]

                # add up the score for the same relation
                if substitute_relations_with_scores.get(substitute_relation) is None:
                    substitute_relations_with_scores[substitute_relation] = relation
                    substitute_relations_with_scores[substitute_relation][
                        "relation"
                    ] = substitute_relation
                else:
                    substitute_relations_with_scores[substitute_relation]["score"] += (
                        relation["score"]
                    )

            retrieve_relations_with_scores = list(
                substitute_relations_with_scores.values()
            )

    elif method_to_score == "embedding":
        entity_relation_embeddings = get_text_embeddings(
            [f"{entity} | {relation}" for relation in all_relations]
        )
        query_embedding = get_text_embeddings([question])[0].reshape(1, -1)
        scores = cosine_similarity(entity_relation_embeddings, query_embedding)
        retrieve_relations_with_scores = [
            {"entity": entity, "relation": relation, "score": score[0]}
            for relation, score in zip(all_relations, scores)
        ]

    if len(retrieve_relations_with_scores) == 0:
        return []
    return retrieve_relations_with_scores


def entity_search(G: nx.Graph, entity: str, relation: str) -> dict[str, dict[str, any]]:
    """
    Return format:

    Ex. 'marc benioff': {'title': 'works at', 'weight': 4, 'head': 'marc benioff', 'tail': 'salesforce'}
    """
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
    elif (
        len(scores) == len(entity_candidates) * 2
        or len(scores) == len(entity_candidates) + 1
    ):
        return scores[: len(entity_candidates)]
    else:
        logging.error(f"Error in scores: {entity_candidates} {scores} ")
        return [1 / len(entity_candidates)] * len(entity_candidates)


def entity_score(
    question: str,
    relation: str,
    entity_candidates: list[str],
    relation_score: float,
    method_to_score: str,
) -> list[float]:
    if method_to_score in ["llm", "llm_embedding"]:
        if len(entity_candidates) == 1:
            scores = [relation_score]
        else:
            user_prompt = f"""Q: {question}
Relation: {relation}
Entities: {entity_candidates}"""
            response = run_llm(
                [
                    {"content": score_entity_candidates_prompt, "role": "system"},
                    {"content": user_prompt, "role": "user"},
                ]
            )
            scores = [
                float(x) * relation_score
                for x in clean_scores(response, entity_candidates)
            ]
    elif method_to_score == "embedding":
        relation_entity_embeddings = get_text_embeddings(
            [f"{relation} {entity}" for entity in entity_candidates]
        )
        query_embedding = get_text_embeddings(question)[0].reshape(1, -1)
        scores = (
            cosine_similarity(relation_entity_embeddings, query_embedding)
            * relation_score
        )
        scores = [score[0] for score in scores]

    return scores


def prune_by_score(all_selected_items: list[dict[str, any]]) -> list[dict[str, any]]:
    all_selected_items.sort(key=lambda x: x["score"], reverse=True)
    return all_selected_items[:TOP_K]


def clean_triples(response: str) -> list[dict[str, any]]:
    patterns = [
        r"{\s*\((?P<triple>[^()]+)\)\s+\(Score:\s+(?P<score>[0-9.]+)\)}",
        r"\*\*\s*\((?P<triple>[^()]+)\)\s+\(Score:\s+(?P<score>[0-9.]+)\):?\*\*",
    ]
    triples = []
    for pattern in patterns:
        for match in re.finditer(pattern, response):
            try:
                triple = match.group("triple").strip()
                score = float(match.group("score"))
                head, relation, tail = triple.split(" | ")
                triples.append(
                    {
                        "head": head.lower(),
                        "title": relation.lower(),
                        "tail": tail.lower(),
                        "score": score,
                    }
                )
            except ValueError:
                logging.error(f"Error in item: {match}")

        if len(triples) > 0:
            break

    return triples


def triple_prune(
    entity: str,
    all_triples: list[dict[str, any]],
    question: str,
    method_to_score: str,
) -> list[dict[str, any]]:
    """
    Return format:

    Ex. {'head': 'salesforce', 'title': 'is described in', 'tail': 'wikipedia', 'score': 0.4}
    """

    if len(all_triples) == 1:
        retrieve_triples_with_scores = all_triples
        retrieve_triples_with_scores[0]["score"] = 1
    else:
        # get triples embeddings
        all_triples_df = pd.DataFrame(all_triples)
        all_triples_df["triple_name"] = (
            all_triples_df["head"]
            + " | "
            + all_triples_df["title"]
            + " | "
            + all_triples_df["tail"]
        )
        all_triples_df.set_index("triple_name", inplace=True)
        all_triples_df["embedding"] = get_text_embeddings(list(all_triples_df.index))

        triple_embeddings = {
            index: embedding
            for index, embedding in zip(
                all_triples_df.index,
                all_triples_df["embedding"],
            )
        }
        triple_embeddings_df = pd.DataFrame(triple_embeddings).T

        if method_to_score in ["llm", "llm_embedding"]:
            batch_size = 50
            substitute_triples_with_scores = []
            for i in range(len(all_triples) // batch_size + 1):
                # ask LLM to choose top k triples
                system_prompt = extract_triple_prompt.format(top_k=TOP_K)

                user_prompt = f"""Q: {question}
Topic Entity: {entity}
Triples: {"; ".join([f"({triple['head']} | {triple['title']} | {triple['tail']})" for triple in all_triples[i * batch_size : (i + 1) * batch_size]])}
A: """
                response = run_llm(
                    [
                        {"content": system_prompt, "role": "system"},
                        {"content": user_prompt, "role": "user"},
                    ]
                )

                # clean the response
                retrieve_triples_with_scores = clean_triples(response)
                logging.info(
                    f"triples retrieved by LLM: {retrieve_triples_with_scores}"
                )

                # check relations retrieved by LLM are in graph
                for triple in retrieve_triples_with_scores:
                    selected_triple = all_triples_df[
                        (all_triples_df["head"] == triple["head"])
                        & (all_triples_df["title"] == triple["title"])
                        & (all_triples_df["tail"] == triple["tail"])
                    ]

                    if selected_triple.shape[0] > 0:
                        logging.info(f"Triple '{triple}' found")
                        substitute_triples_with_scores.append(triple)
                    else:
                        if method_to_score == "llm":
                            logging.error(f"Triple '{triple}' not found")
                        elif method_to_score == "llm_embedding":
                            substitute_triple = retrieve_top_k_similarity(
                                triple_embeddings_df,
                                get_text_embeddings(
                                    [
                                        f"{triple['head']} | {triple['title']} | {triple['tail']}"
                                    ]
                                )[0],
                                1,
                            )[0]
                            substitute_triples_with_scores.append(
                                {
                                    "head": substitute_triple.split(" | ")[0],
                                    "title": substitute_triple.split(" | ")[1],
                                    "tail": substitute_triple.split(" | ")[2],
                                    "score": triple["score"],
                                }
                            )
                            logging.info(
                                f"Triple '{triple}' not found -> change to {substitute_triple}"
                            )
            retrieve_triples_with_scores = substitute_triples_with_scores
            if len(retrieve_triples_with_scores) > TOP_K:
                retrieve_triples_with_scores = triple_prune(
                    entity,
                    retrieve_triples_with_scores,
                    question,
                    method_to_score,
                )
        elif method_to_score == "embedding":
            query_embedding = get_text_embeddings([question])[0].reshape(1, -1)
            scores = cosine_similarity(triple_embeddings_df, query_embedding)
            retrieve_triples_with_scores = [
                {
                    "head": index.split(" | ")[0],
                    "title": index.split(" | ")[1],
                    "tail": index.split(" | ")[2],
                    "score": score[0],
                }
                for index, score in zip(triple_embeddings_df.index, scores)
            ]

        if len(retrieve_triples_with_scores) == 0:
            return []
    return retrieve_triples_with_scores


def get_new_topic_entities(
    important_edges: list[dict[str, any]], topic_entities: list[str]
) -> list[str]:
    new_topic_entities = set()
    for edge in important_edges:
        if edge["head"] not in topic_entities:
            new_topic_entities.add(edge["head"])
        if edge["tail"] not in topic_entities:
            new_topic_entities.add(edge["tail"])
    return list(new_topic_entities)


def reasoning(
    question: str, cluster_chain_of_entities: list[list[dict[str, any]]]
) -> str:
    chain_prompt = "\n".join(
        [
            f"{triple['head']}, {triple['title']}, {triple['tail']}"
            for sublist in cluster_chain_of_entities
            for triple in sublist
        ]
    )
    user_prompt = f"""Q: {question}
Knowledge Triplets:\n{chain_prompt}
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
    path: str, cluster_chain_of_entities: list[list[dict[str, any]]], round_count: int
) -> None:
    all_triples = [[]]
    for cluster in cluster_chain_of_entities:
        for triple in cluster:
            all_triples[0].append(
                {
                    "subject": triple["head"],
                    "relationship": triple["title"],
                    "object": triple["tail"],
                }
            )

    tempG = draw_graph(all_triples, labels_entities)
    nt = Network(height="750px", width="100%")
    nt.from_nx(tempG)
    nt.force_atlas_2based(central_gravity=0.015, gravity=-31)
    nt.save_graph(f"{path}/{REASONING_DIR_NAME}/graph{round_count}.html")


def generate_answer(
    question: str, cluster_chain_of_entities: list[list[dict[str, any]]]
) -> tuple[str, str]:
    chain_prompt = "\n".join(
        [
            f"{triple['head']}, {triple['title']}, {triple['tail']}"
            for sublist in cluster_chain_of_entities
            for triple in sublist
        ]
    )
    user_prompt = f"""Q: {question}
Knowledge Triplets:\n{chain_prompt}
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

TOP_K = 5

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", type=str, required=True)
    arg_parser.add_argument("--query", type=str, required=True)
    arg_parser.add_argument(
        "--method_for_query_entity",
        type=str,
        choices=["ner", "ner_embedding", "embedding"],
        default="ner",
    )
    arg_parser.add_argument(
        "--method_to_score",
        type=str,
        choices=["llm", "llm_embedding", "embedding"],
        default="llm",
    )
    arg_parser.add_argument(
        "--traverse_step",
        type=str,
        choices=["entity_relation", "triple"],
        default="entity_relation",
    )
    args = arg_parser.parse_args()

    # path
    DATA_PATH = args.path
    ARG_ID = (
        f"{args.method_for_query_entity}__{args.method_to_score}__{args.traverse_step}"
    )
    REASONING_DIR_NAME = f"reasoning_path__{ARG_ID}"
    if not os.path.exists(f"{DATA_PATH}/construction"):
        exit()
    if not os.path.exists(f"{DATA_PATH}/additional"):
        os.makedirs(f"{DATA_PATH}/additional")
    if not os.path.exists(f"{DATA_PATH}/{REASONING_DIR_NAME}"):
        os.makedirs(f"{DATA_PATH}/{REASONING_DIR_NAME}")

    # logging
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"{DATA_PATH}/log/{REASONING_DIR_NAME}.log",
                mode="w",
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )
    CONSTRUCTION_DIR = f"{DATA_PATH}/construction"

    # === Query ===
    question = args.query

    # === LLM only ===
    # response_without_explored_paths = generate_without_explored_paths(question)

    # === Create Graph ===
    ENTITY_LIST_PATHS = []
    CHUNKS_ENTITIES_PATHS = []
    TRIPLES_PATHS = []
    ERRORS_PATHS = []

    for file in os.listdir(CONSTRUCTION_DIR):
        if file.startswith("entities"):
            ENTITY_LIST_PATHS.append(f"{CONSTRUCTION_DIR}/{file}")
        elif file.startswith("chunks_entities"):
            CHUNKS_ENTITIES_PATHS.append(f"{CONSTRUCTION_DIR}/{file}")
        elif file.startswith("chunks_triples"):
            TRIPLES_PATHS.append(f"{CONSTRUCTION_DIR}/{file}")
        elif file.startswith("errors"):
            ERRORS_PATHS.append(f"{CONSTRUCTION_DIR}/{file}")

    # entities
    entity_list = []
    chunks_entities = []
    for entity_list_path, chunks_entities_path in zip(
        ENTITY_LIST_PATHS, CHUNKS_ENTITIES_PATHS
    ):
        entity_list_part, chunks_entities_part = extract_entities(
            [], [], entity_list_path, chunks_entities_path
        )
        entity_list.extend(entity_list_part)
        chunks_entities.extend(chunks_entities_part)
    logging.info(f"# of entities: {len(entity_list)}")

    labels_entities = classify_entities(entity_list, LABELS)
    logging.info({label: len(entities) for label, entities in labels_entities.items()})

    # triples
    chunks_triples = []
    for triples_path, errors_path in zip(TRIPLES_PATHS, ERRORS_PATHS):
        chunks_triples_part, _ = extract_triples([], [], triples_path, errors_path)
        chunks_triples.extend(chunks_triples_part)
    logging.info(f"# of triples: {sum(len(triples) for triples in chunks_triples)}")

    # graph
    GRAPH_PATH = f"{CONSTRUCTION_DIR}/graph.html"
    G = draw_graph(chunks_triples, labels_entities)

    # entity embeddings
    ENTITY_EMBEDDINGS_PATH = f"{DATA_PATH}/additional/entity_embeddings.pkl"
    entity_embeddings = get_all_entity_embeddings(G, ENTITY_EMBEDDINGS_PATH)
    entity_embeddings_df = pd.DataFrame(entity_embeddings).T

    # === ToG ===
    # query topic entities
    topic_entities = []
    if args.method_for_query_entity == "ner":
        _, chunks_entities = extract_entities([question], LABELS)
        ner_entities = chunks_entities[0]
        logging.info(f"query entities extracted from NER: {ner_entities}")
        topic_entities = [
            ner_entity.lower() for ner_entity in ner_entities if ner_entity in G.nodes
        ]
    elif args.method_for_query_entity == "ner_embedding":
        # ner
        _, chunks_entities = extract_entities([question], LABELS)
        ner_entities = chunks_entities[0]
        logging.info(f"query entities extracted from NER: {ner_entities}")

        # embedding
        topic_entities = []
        ner_entity_embeddings = get_text_embeddings(ner_entities)
        for idx, ner_entity_embedding in enumerate(ner_entity_embeddings):
            topic_entities.append(
                retrieve_top_k_similarity(
                    entity_embeddings_df, ner_entity_embedding, 1
                )[0]
            )
    elif args.method_for_query_entity == "embedding":
        query_embedding = get_text_embeddings([question])[0]
        topic_entities.extend(
            retrieve_top_k_similarity(entity_embeddings_df, query_embedding, TOP_K)
        )
    logging.info(f"Topic entities: {topic_entities}")

    # traversal on graph
    done = False
    round_count = 1
    cluster_chain_of_entities = []

    while not done:
        logging.info(f"\n===== Round {round_count}: start from {topic_entities} =====")

        if args.traverse_step == "entity_relation":
            # === relation ===
            current_entity_relations_list = []

            for idx, entity in enumerate(topic_entities):
                try:
                    # find all adjacent relations not retrieved yet
                    all_unused_adjacent_triples = unused_adjacent_triple_search(
                        G, entity, cluster_chain_of_entities
                    )
                    if len(all_unused_adjacent_triples) == 0:
                        logging.info(f"{entity}: 0 relation selected")
                        continue

                    # score the relations
                    retrieve_relations_with_scores = relation_prune(
                        entity,
                        all_unused_adjacent_triples,
                        question,
                        args.method_to_score,
                    )
                    current_entity_relations_list.extend(retrieve_relations_with_scores)

                    logging.info(
                        f"{entity}: {len(retrieve_relations_with_scores)} relation selected\n{retrieve_relations_with_scores}"
                    )
                except KeyError:
                    logging.error(f"{entity}: not found")

            logging.info(
                f"\ncurrent_entity_relations_list:\n{current_entity_relations_list}"
            )

            if len(current_entity_relations_list) == 0:
                break

            # === entity ===
            all_selected_edges = []

            for entity_relation in current_entity_relations_list:
                entity = entity_relation["entity"]
                relation = entity_relation["relation"]
                score = entity_relation["score"]

                # find all adjacent nodes with the relation
                adjacent_edges_with_relation = entity_search(G, entity, relation)
                entity_candidates = list(adjacent_edges_with_relation.keys())

                if len(entity_candidates) == 0:
                    continue

                # score the entities
                scores = entity_score(
                    question,
                    entity_relation["relation"],
                    entity_candidates,
                    entity_relation["score"],
                    args.method_to_score,
                )

                # add score to edges
                for idx, entity in enumerate(entity_candidates):
                    adjacent_edges_with_relation[entity]["score"] = scores[idx]

                all_selected_edges.extend(list(adjacent_edges_with_relation.values()))

            logging.info(f"\nall_selected_edges:\n{all_selected_edges}")

            important_edges = prune_by_score(all_selected_edges)
            logging.info(f"\nimportant_edges:\n{important_edges}")
        elif args.traverse_step == "triple":
            all_selected_edges = []

            for idx, entity in enumerate(topic_entities):
                # find all adjacent relations not retrieved yet
                all_unused_adjacent_triples = unused_adjacent_triple_search(
                    G, entity, cluster_chain_of_entities
                )
                if len(all_unused_adjacent_triples) == 0:
                    logging.info(f"{entity}: 0 triple selected")
                    continue

                # score the triples
                retrieve_triples_with_scores = triple_prune(
                    entity,
                    all_unused_adjacent_triples,
                    question,
                    args.method_to_score,
                )
                all_selected_edges.extend(retrieve_triples_with_scores)

                logging.info(
                    f"{entity}: {len(retrieve_triples_with_scores)} triple selected\n{retrieve_triples_with_scores}"
                )

            logging.info(f"\nall_selected_edges:\n{all_selected_edges}")

            important_edges = prune_by_score(all_selected_edges)
            logging.info(f"\nimportant_edges:\n{important_edges}")

        # update
        topic_entities = get_new_topic_entities(important_edges, topic_entities)

        if len(important_edges) > 0:
            logging.info("\nSelected edges:")
            show_adjacent_relations(important_edges)

            for idx in range(len(important_edges)):
                try:
                    del important_edges[idx]["score"]
                except KeyError:
                    logging.error(f"Score not found in {important_edges[idx]}")
            cluster_chain_of_entities.append(important_edges)

            save_current_cluster_chain(
                DATA_PATH, cluster_chain_of_entities, round_count
            )

            response = reasoning(question, cluster_chain_of_entities)
            done = extract_answer(response)
        else:
            logging.info("No edges selected")
            done = True

        if round_count == 20:
            logging.info("Max round reached")
            done = True

        round_count += 1

    # === Answer ===
    prompt, response = generate_answer(question, cluster_chain_of_entities)
    RESULT_PATH = f"{DATA_PATH}/log/result__{ARG_ID}.txt"
    with open(RESULT_PATH, "w") as f:
        json.dump(
            {
                "query": question,
                "retrieved data": cluster_chain_of_entities,
                "answer": response,
            },
            f,
            indent=4,
        )
