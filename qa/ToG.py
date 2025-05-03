import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional
import ast
import logging
from datetime import datetime
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from lm import LLM, NER
from graph import KG
from qa.QA import QA
from utils.similarity import get_fuzzy_best_match, get_cosine_similarity_best_match


class ToG(QA):
    extract_topic_entities_prompt = """Please retrieve {top_k} topic entities that contribute to answering the following question, and rate their contribution on a scale from 0 to 1. The scores must sum to 1.
Q: Name the president of the country whose main spoken language was Brahui in 1980?
A:
1. {{Brahui Language (Score: 0.4)}}: This is the key clue in the question and helps identify the country being referred to.
2. {{Pakistan (Score: 0.3)}}: The country where Brahui is primarily spoken, necessary for determining its president in 1980.
3. {{President of Pakistan (Score: 0.3)}}: Represents the actual target answer type, and is conceptually central to the question.
"""

    prune_relation_prompt = """Please retrieve {top_k} relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of {top_k} relations is 1).
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: main country; language family; ISO 639-3 code; parent language; writing system; language class; countries spoken in; prominent type; related document; equivalent instances; local name; region
A: 
1. {{main country (Score: 0.4))}}: This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
2. {{countries spoken in (Score: 0.3)}}: This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.
3. {{parent language (Score: 0.2)}}: This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.
"""

    prune_entity_prompt = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).
Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Relation: producer
Entities: The Resident; So Undercover; Let Me In; Begin Again; The Quiet Ones; A Walk Among the Tombstones
A:
1. {{So Undercover (Score: 1.0)}}: This entity is highly relevant as it is the movie that both features Miley Cyrus and is produced by Tobin Armbrust, directly answering the question.
2. {{The Resident (Score: 0.0)}}: This entity is not relevant because it does not feature Miley Cyrus.
3. {{Let Me In (Score: 0.0)}}: This entity is not relevant because it does not feature Miley Cyrus nor was it produced by Tobin Armbrust.
4. {{Begin Again (Score: 0.0)}}: This entity is not relevant to the question.
5. {{The Quiet Ones (Score: 0.0)}}: This entity is not relevant to the question.
6. {{A Walk Among the Tombstones (Score: 0.0)}}: This entity is not relevant to the question.
"""

    path_select_prompt = """Please retrieve {top_k} triples (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of {top_k} relations is 1). Format your answer with parentheses and curly braces like {{(entity1 | relation | entity2) (Score: x.x)}}.
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Triples: (Brahui Language | main country | Pakistan); (brh | corresponds to | Brahui Language); (Arabic script | used by | Brahui Language); (Brahui Language | countries spoken in | Pakistan); (Brahui Language | parent language | Dravidian); (Brahui Language | region | Balochistan)
A: 
1. {{(Brahui Language | main country | Pakistan) (Score: 0.5)}}: This triple remains the most relevant since the focus of the question is on identifying the country where Brahui was the main spoken language in 1980, directly tied to the question.
2. {{(Brahui Language | countries spoken in | Pakistan) (Score: 0.3)}}: This triple confirms and reinforces that Brahui is spoken in Pakistan, supporting the primary answer and narrowing the search.
3. {{(Brahui Language | region | Balochistan) (Score: 0.2)}}: I would prioritize this triple over "parent language" because it adds geographical context that is directly relevant to identifying Pakistan. Understanding that Balochistan is a key region where Brahui is spoken enhances the overall relevance.
"""

    prompt_evaluate_with_subgraphs = """Given a question and the associated retrieved knowledge graphs, you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No). Please format your answer with curly braces: {Yes} or {No}.
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Graph: 
Taste cannot be controlled by law
└── Thomas Jefferson ─── (quotation from)
A: {No}. Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.

Q: The artist nominated for The Long Winter lived where?
Knowledge Graph: 
Unknown-Entity
└── De Smet ─── (lived in)

Laura Ingalls Wilder
└── Unknown-Entity ─── (lived in)

The Long Winter
└── Laura Ingalls Wilder ─── (written by)
A: {Yes}. Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {De Smet}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Graph: 
Steve Bisciotti
├── Baltimore Ravens ─── (owns the team)
├── Baltimore Ravens ─── (is the owner of the professional sports team)
└── Allegis Group ─── (is the founder of)
A: {No}. Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Graph: 
Rift Valley Province
├── Kenya ─── (is an administrative division in)
|   └── Kenyan shilling ─── (used as its currency)
├── UnName_Entity ─── (has a geographic location)
└── UnName_Entity ─── (is part of a mailing address region of)
A: {Yes}. Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {Kenyan shilling}.

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Graph: 
National Anthem of Bolivia
├── UnName_Entity ─── (is an anthem of)
├── Leopoldo Benedetto Vincenti ─── (was composed by)
└── José Ignacio de Sanjinés ─── (lyrics were wrote by)

Bolivia
└── UnName_Entity ─── (has anthem)

UnName_Entity
└── Bolivia ─── (is an anthem of)
A: {No}. Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.
"""

    generate_new_query_with_triples_prompt = """Given a question and a set of knowledge graph triplets (entity, relation, entity), since the current information is insufficient to answer the question, generate one follow-up question to help the user gather more relevant details.
Respond with only a single JSON string with an "query" field which is the follow-up question."""

    generate_new_query_with_subgraphs_prompt = """Given a question and the associated retrieved knowledge graphs, since the current information is insufficient to answer the question, generate one follow-up question to help the user gather more relevant details.
Respond with only a single JSON string with an "query" field which is the follow-up question."""

    answer_prompt = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge.
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Graph: 
Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
A: Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.

Q: The artist nominated for The Long Winter lived where?
Knowledge Graph: 
The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
A: Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {De Smet}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Graph: 
Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
A: Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Graph: 
Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
A: Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {Kenyan shilling}.

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Graph: 
National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
A: Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.
"""

    def __init__(
        self,
        kg: KG,
        top_n: int = 3,
        query_topic_entities_select_method: str = "llm",
        scoring_method: str = "llm",
        path_select_method: str = "score",
        generate_new_query: bool = False,
        integration_style: str = "triples",
        integrate_reasoning_response: bool = False,
        max_iterations: int = 10,
        llm_verbose: bool = False,
        logger: logging.Logger = logging.getLogger(),
    ):
        """
        - `kg`
        - `top_n`: for pruning
        - `query_topic_entities_select_method`: for query -> entities; [llm], ner
        - `scoring_method`: for pruning relations/entities; [llm], embedding
        - `path_select_method`: for pruning retrieved triples; [score], embedding, llm_w_history, llm_wo_history
        - `generate_new_query`: for generating new query after a round; [False], True
        - `integration_style`: for how to integrate the retrieved triples into LLM; [triples], paths, trees
        - `integrate_reasoning_response`: for whether to add reasoning response from LLM into answering step; [False], True
        """
        self.logger = logger
        self.llm_model = LLM(verbose=llm_verbose, logger=self.logger)
        self.kg = kg
        self.top_n = top_n
        self.query_topic_entities_select_method = query_topic_entities_select_method
        self.scoring_method = scoring_method
        if scoring_method == "embedding":
            self.logger.info("Get entities & relations embedding...")
            self._get_all_embeddings()
        self.path_select_method = path_select_method
        self.generate_new_query = generate_new_query
        self.integration_style = integration_style
        self.integrate_reasoning_response = integrate_reasoning_response
        self.max_iterations = max_iterations

    def _get_all_embeddings(self):
        self.kg.entities_emb = self.llm_model.embed(list(self.kg.entities.keys()))
        unique_relations = list(
            {relation["label"] for relation in self.kg.relations.values()}
        )
        self.kg.relations_emb = self.llm_model.embed(unique_relations)

    def retrieve(self, query: str, root_path: Optional[str] = None) -> dict[str, any]:
        used_kg = KG()

        # 1. initialization
        current_query = query
        query_topic_entities_score = self._get_topic_entities(current_query, used_kg)
        topic_entities_score = query_topic_entities_score.copy()
        round_count = 0
        data_sufficient = False
        early_stop = False

        while not data_sufficient and not early_stop:
            round_count += 1
            self.logger.info(f"=== Round {round_count}: {current_query} ===")
            self.logger.info(f"=> Topic entities: {topic_entities_score}")

            # 2. exploration
            triples_score = {}
            triples_topic = {}
            for topic_entity, topic_entity_score in topic_entities_score.items():
                # 2-1. relation exploration
                relations = self._relation_search(used_kg, topic_entity)
                relations_score = self._relation_prune(
                    current_query, topic_entity, relations
                )
                self.logger.info(f"    {topic_entity} => Relations: {relations_score}")

                # 2-2. entity exploration
                for relation, relation_score in relations_score.items():
                    entities = self._entity_search(used_kg, topic_entity, relation)
                    entities_score = self._entity_prune(
                        current_query, relation, entities
                    )
                    self.logger.info(
                        f"        {relation} => Entities: {entities_score}"
                    )

                    # update triples score
                    for entity, entity_score in entities_score.items():
                        relation_info = self.kg.relations.get(
                            (topic_entity, relation, entity), None
                        )
                        if relation_info is None:
                            relation_info = self.kg.relations.get(
                                (entity, relation, topic_entity), None
                            )
                        triple = (
                            relation_info["subject"],
                            relation,
                            relation_info["object"],
                        )
                        triples_score[triple] = triples_score.get(triple, 0) + (
                            topic_entity_score * relation_score * entity_score
                        )
                        triples_topic[triple] = triples_topic.get(triple, set())
                        triples_topic[triple].add(topic_entity)

            # run again when nothing retrieved
            if len(triples_score) == 0:
                topic_entities_score = self._update_topic_entities(
                    triples_score, triples_topic
                )
                if len(topic_entities_score) == 0:
                    topic_entities_score = self._get_topic_entities(
                        current_query, used_kg
                    )
                round_count -= 1
                continue

            # 3. pruning
            selected_triples_score = self._triple_prune(query, triples_score, used_kg)
            for triple in selected_triples_score:
                self._add_triple_to_used_kg(used_kg, triple, round_count)

            # show progress
            self.logger.info("=> Triples:")
            for triple, score in sorted(
                list(selected_triples_score.items()), key=lambda x: x[1], reverse=True
            ):
                self.logger.info(f"{triple}: {score:.2f}")

            self.logger.info("=> Retrieved subgraph:")
            retrieved_subgraph = used_kg.format_as_trees(["round"])
            self.logger.info("\n" + retrieved_subgraph)

            if root_path is not None:
                if not os.path.exists(root_path):
                    os.makedirs(root_path)
                used_kg.save_graph(f"{root_path}/used_graph_{round_count}.html")

            if round_count == self.max_iterations:
                early_stop = True

            # 4. reasoning
            data_sufficient, reasoning_response = self._reasoning(query, used_kg)

            if not data_sufficient and not early_stop:
                # update for next round
                if self.generate_new_query:
                    new_query = self._generate_query(query, used_kg)

                if self.generate_new_query and new_query != current_query:
                    current_query = new_query
                    topic_entities_score = self._get_topic_entities(
                        current_query, used_kg
                    )
                else:
                    topic_entities_score = self._update_topic_entities(
                        selected_triples_score, triples_topic
                    )
                    if len(topic_entities_score) == 0:
                        topic_entities_score = self._get_topic_entities(
                            current_query, used_kg
                        )

                if len(topic_entities_score) == 0:
                    early_stop = True

        # collect information
        retrieval_records = {
            "settings": vars(self),
            "used_kg": used_kg,
            "round_count": round_count,
            "early_stop": early_stop,
            "reasoning_response": reasoning_response,
        }

        # save the final retrieved graph
        if root_path is not None:
            self._save_reasoning_graph(used_kg, root_path)
        return retrieval_records

    def _get_topic_entities(self, query: str, used_kg: KG) -> dict[str, float]:
        if self.query_topic_entities_select_method == "llm":
            # ask LLM to extract topic entities
            response = self.llm_model.chat(
                [
                    {
                        "role": "system",
                        "content": ToG.extract_topic_entities_prompt.format(
                            top_k=self.top_n
                        ),
                    },
                    {"role": "user", "content": f"Q: {query}"},
                ]
            )
            results = self.llm_model.parse_response(
                response,
                r"(?:\{|\*{2})(.*?)\s*\(Score:\s*([0-9.]+)\)[ :]*(?:\}|\*{2})",
                {"entity": str, "score": float},
            )
        elif self.query_topic_entities_select_method == "ner":
            ner_model = NER()
            results = ner_model.extract_entities(query)
            results = [
                {"entity": entity, "score": 1 / len(results)} for entity in results
            ]

        # match with KG nodes
        unused_entities = self._get_unused_entities(used_kg)
        topic_entities = self._match_score("entity", results, unused_entities)

        return topic_entities

    def _get_unused_entities(self, used_kg: KG) -> list[str]:
        unused_entities = set()
        for entity in self.kg.entities:
            if entity not in used_kg:
                unused_entities.add(entity)
            else:
                neighbors = self.kg[entity].keys()
                for neighbor in neighbors:
                    if used_kg[entity].get(neighbor) is None or len(
                        self.kg[entity][neighbor]
                    ) != len(used_kg[entity][neighbor]):
                        unused_entities.add(entity)
                        break
        return list(unused_entities)

    def _match_score(
        self,
        label_name: str,
        score_results: list[dict[str, any]],
        match_targets: list[str],
        threshold: float = None,
    ) -> dict[str, float]:
        """
        label_name: relation or entity

        score_results format: top_n with score

        [{'relation': 'directed', 'score': 0.6},
        {'relation': 'collaborated with', 'score': 0.3},
        {'relation': 'was directed by', 'score': 0.1}]

        return format: top_n matched with score

        {'directed': 0.6, 'collaborated with': 0.3, 'was directed by': 0.1}
        """
        # match with KG nodes or edges
        matched_results = {
            result[label_name]: result["score"]
            for result in score_results
            if result[label_name] in match_targets
        }

        for result in score_results:
            match_subject = result[label_name].lower()
            score = result["score"]
            if match_subject not in matched_results:
                if threshold:
                    best_match = get_fuzzy_best_match(
                        match_subject, match_targets, threshold=threshold
                    )
                else:
                    best_match = get_fuzzy_best_match(match_subject, match_targets)
                best_match = list(best_match.keys())
                if len(best_match) != 0:
                    matched_results[best_match[0]] = (
                        matched_results.get(best_match[0], 0) + score
                    )

        return matched_results

    def _relation_search(self, used_kg: KG, entity: str) -> list[str]:
        neighbor_relations = set()
        for neighbor_entity, edges in self.kg[entity].items():
            for edge in edges.values():
                if not used_kg.has_edge(entity, neighbor_entity, edge["label"]):
                    neighbor_relations.add(edge["label"])

        return list(neighbor_relations)

    def _relation_prune(
        self, query: str, entity: str, relations: list[str]
    ) -> dict[str, float]:
        """return format: top_n relation with score

        {'directed': 0.6, 'collaborated with': 0.3, 'was directed by': 0.1}
        """
        if len(relations) == 0:
            return {}
        if len(relations) == 1:
            return {relations[0]: 1.0}

        if self.scoring_method == "llm":
            # ask LLM to prune relations
            response = self.llm_model.chat(
                [
                    {
                        "role": "system",
                        "content": ToG.prune_relation_prompt.format(top_k=self.top_n),
                    },
                    {
                        "role": "user",
                        "content": f"""Q: {query}
Topic Entity: {entity}
Relations: {"; ".join(relations)}
A: """,
                    },
                ]
            )
            results = self.llm_model.parse_response(
                response,
                r"(?:\{|\*{2})(.*?)\s*\(Score:\s*([0-9.]+)\)[ :]*(?:\}|\*{2})",
                {"relation": str, "score": float},
            )
        elif self.scoring_method == "embedding":
            # get best match with KG entities through embeddings
            query_emb = self.llm_model.embed([query])[query]
            best_match_score = get_cosine_similarity_best_match(
                query_emb,
                {relation: self.kg.relations_emb[relation] for relation in relations},
                threshold=0,
                top_n=self.top_n,
            )
            results = [
                {"relation": match, "score": score}
                for match, score in best_match_score.items()
            ]

        # match with KG relations
        match_relations = self._match_score("relation", results, relations)

        return match_relations

    def _entity_search(self, used_kg: KG, entity: str, relation: str) -> list[str]:
        neighbor_entities = set()
        for neighbor_entity, edges in self.kg[entity].items():
            for edge in edges.values():
                if edge["label"] == relation and not used_kg.has_edge(
                    edge["subject"], edge["object"], edge["label"]
                ):
                    neighbor_entities.add(neighbor_entity)

        return list(neighbor_entities)

    def _entity_prune(
        self, query: str, relation: str, entities: list[str]
    ) -> dict[str, float]:
        """return format: top_n entity with score

        {'Tom Hanks': 0.6, 'Steven Spielberg': 0.3, 'Jaws': 0.1}
        """
        if len(entities) == 0:
            return {}
        if len(entities) == 1:
            return {entities[0]: 1.0}

        if self.scoring_method == "llm":
            # ask LLM to prune entities
            response = self.llm_model.chat(
                [
                    {
                        "role": "system",
                        "content": ToG.prune_entity_prompt.format(top_k=self.top_n),
                    },
                    {
                        "role": "user",
                        "content": f"""Q: {query}
Relation: {relation}
Entities: {"; ".join(entities)}
A: """,
                    },
                ]
            )
            results = self.llm_model.parse_response(
                response,
                r"(?:\{|\*{2})(.*?)\s*\(Score:\s*([0-9.]+)\)[ :]*]*(?:\}|\*{2})",
                {"entity": str, "score": float},
            )
        elif self.scoring_method == "embedding":
            # get best match with KG entities through embeddings
            query_emb = self.llm_model.embed([query])[query]
            best_match_score = get_cosine_similarity_best_match(
                query_emb,
                {entity: self.kg.entities_emb[entity] for entity in entities},
                threshold=0,
                top_n=self.top_n,
            )
            results = [
                {"entity": match, "score": score}
                for match, score in best_match_score.items()
            ]

        # match with KG entities
        match_entities = self._match_score("entity", results, entities)

        return match_entities

    def _triple_prune(
        self,
        query: str,
        triples_score: dict[tuple[str, str, str], float],
        used_kg: KG,
    ) -> dict[tuple[str, str, str], float]:
        """- `path_select_method`: for pruning retrieved triples; [score], embedding, llm_w_history, llm_wo_history"""
        if len(triples_score) <= 1:
            return {triple: 1 for triple in triples_score}

        if self.path_select_method == "embedding":
            query_emb = self.llm_model.embed([query])[query]

            # Generate embeddings for triples
            triples_str_list = [" ".join(triple) for triple in triples_score]
            embeddings = self.llm_model.embed(triples_str_list)
            triples_emb = {
                triple: embeddings[triple_str]
                for triple, triple_str in zip(triples_score, triples_str_list)
            }

            triples_score = get_cosine_similarity_best_match(
                query_emb,
                triples_emb,
                threshold=0,
                top_n=self.top_n,
            )
        elif self.path_select_method in ["llm_wo_history", "llm_w_history"]:
            formatted_triples = ""
            for triple in triples_score:
                formatted_triples += f"({' | '.join(triple)})" + "; "

            # ask LLM to prune triples
            if self.path_select_method == "llm_wo_history":
                user_prompt = f"""Q: {query}
Triples: {formatted_triples}
A: """
            else:
                retrieved_data = self._format_retrieved_data(used_kg)
                user_prompt = f"""Q: {query}
Knowledge Graph:
{retrieved_data}

The above knowledge graph shows what is already known. Please choose and score the triples below that are missing from the graph, but potentially useful for answering the question.

Triples: {formatted_triples}
A: """

            response = self.llm_model.chat(
                [
                    {
                        "role": "system",
                        "content": ToG.path_select_prompt.format(top_k=self.top_n),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ]
            )
            results = self.llm_model.parse_response(
                response,
                r"\{(.*?)\s*\(Score:\s*([0-9.]+)\)\s*\}",
                {"triple": str, "score": float},
            )
            triples_str_list = [" | ".join(triple) for triple in triples_score]
            triples_score = {}
            for triple_str, score in self._match_score(
                "triple", results, triples_str_list
            ).items():
                triple = tuple([element.strip() for element in triple_str.split(" | ")])
                triples_score[triple] = score

        # select top_n triples with score
        selected_triples = sorted(
            triples_score.items(),
            key=lambda x: x[1],
            reverse=True,
        )[: self.top_n]
        return {triple: score for triple, score in selected_triples if score > 0}

    def _add_triple_to_used_kg(
        self,
        used_kg: KG,
        triple: tuple[str, str, str],
        round_count: int,
    ) -> None:
        used_kg.add_node(triple[0], **self.kg.entities[triple[0]])
        used_kg.add_node(triple[2], **self.kg.entities[triple[2]])
        used_kg.add_edge(
            triple[0], triple[2], **self.kg.relations[triple], round=round_count
        )

    def _format_retrieved_data(self, used_kg: KG) -> str:
        if self.integration_style == "triples":
            retrieved_data = used_kg.format_as_triples("round")
        elif self.integration_style == "paths":
            retrieved_data = used_kg.format_as_paths("round")
        elif self.integration_style == "trees":
            retrieved_data = used_kg.format_as_trees()
        return retrieved_data

    def _generate_query(self, query: str, used_kg: KG) -> str:
        retrieved_data = self._format_retrieved_data(used_kg)

        if self.integration_style in ["triples", "paths"]:
            response = self.llm_model.chat(
                [
                    {
                        "role": "system",
                        "content": ToG.generate_new_query_with_triples_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Q: {query}\nKnowledge Graph:\n{retrieved_data}\nNew query: ",
                    },
                ]
            )
        elif self.integration_style == "trees":
            response = self.llm_model.chat(
                [
                    {
                        "role": "system",
                        "content": ToG.generate_new_query_with_subgraphs_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Q: {query}\nKnowledge Graph:\n{retrieved_data}\nNew query: ",
                    },
                ]
            )
        results = self.llm_model.parse_response(
            response,
            r"(\{\n.*?\n\})",
            {"query": str},
        )
        if len(results) == 0:
            return query

        result = results[0]["query"]
        new_query = ast.literal_eval(result)["query"]
        return new_query

    def _update_topic_entities(
        self,
        selected_triples_score: dict[tuple[str, str, str], float],
        triples_topic: dict[tuple[str, str, str], set[str]],
    ) -> dict[str, float]:
        new_topic_entities_score = {}
        for triple, score in selected_triples_score.items():
            topic_entities = list(triples_topic[triple])
            if len(topic_entities) == 2:
                continue
            elif len(topic_entities) == 1:
                subject_entity, _, object_entity = triple
                if topic_entities[0] == subject_entity:
                    new_topic_entities_score[object_entity] = score
                else:
                    new_topic_entities_score[subject_entity] = score

        total_score = sum(new_topic_entities_score.values())
        for entity, score in new_topic_entities_score.items():
            new_topic_entities_score[entity] = round(score / total_score, 2)
        return new_topic_entities_score

    def _reasoning(self, query: str, used_kg: KG) -> tuple[bool, str]:
        retrieved_data = self._format_retrieved_data(used_kg)

        # ask LLM to check whether the retrieved data is sufficient to answer the question
        user_prompt = f"""Q: {query}
Knowledge Graph:\n{retrieved_data}
A: """
        response = self.llm_model.chat(
            [
                {"role": "system", "content": ToG.prompt_evaluate_with_subgraphs},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        )

        # parse response
        results = self.llm_model.parse_response(
            response.lower(),
            r"\{([yes|no]*)\}",
            {"answer": str},
        )

        data_sufficient = False
        if len(results) > 0 and results[0]["answer"] == "yes":
            data_sufficient = True

        return data_sufficient, response

    def _save_reasoning_graph(self, used_kg: KG, root_path: str) -> None:
        # copy graph
        graph = copy.deepcopy(self.kg)
        for node in graph.nodes():
            del graph.nodes[node]["color"]

        # color map
        max_round = max(
            [
                info["round"]
                for _, _, info in used_kg.edges(data=True)
                if "round" in info
            ]
        )
        cmap = plt.get_cmap("turbo")
        gradient = np.linspace(0, 1, max_round)
        colors = [mcolors.to_hex(cmap(val)) for val in gradient]

        # apply color
        for node1, node2, info in used_kg.edges(data=True):
            if "round" in info:
                graph.nodes[node1]["color"] = min(
                    graph.nodes[node1].get("color", max_round), info["round"] - 1
                )
                graph.nodes[node2]["color"] = min(
                    graph.nodes[node2].get("color", max_round), info["round"] - 1
                )
                for key in graph[node1][node2]:
                    graph.edges[node1, node2, key]["color"] = min(
                        graph.edges[node1, node2, key].get("color", max_round),
                        info["round"] - 1,
                    )
        for node in graph.nodes():
            color_id = graph.nodes[node].get("color", None)
            if isinstance(color_id, int):
                graph.nodes[node]["color"] = colors[color_id]
        for node1, node2, _ in graph.edges(data=True):
            for key in graph[node1][node2]:
                color_id = graph.edges[node1, node2, key].get("color", None)
                if isinstance(color_id, int):
                    graph.edges[node1, node2, key]["color"] = colors[color_id]

        # save graph
        graph.save_graph(f"{root_path}/final.html")

    def answer(self, query: str, root_path: Optional[str] = None) -> str:
        start_time = datetime.now()
        retrieval_records = self.retrieve(query, root_path)
        retrieved_data = self._format_retrieved_data(retrieval_records["used_kg"])

        if self.integrate_reasoning_response and not retrieval_records["early_stop"]:
            user_prompt = f"""Q: {query}
Knowledge Graph:
{retrieved_data}

Notes:
{retrieval_records["reasoning_response"]}

A: """
        else:
            user_prompt = f"""Q: {query}
Knowledge Graph:
{retrieved_data}

A: """
        response = self.llm_model.chat(
            [
                {"role": "system", "content": ToG.answer_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        )
        end_time = datetime.now()

        retrieval_records["settings"].pop("logger")
        retrieval_records["settings"].pop("llm_model")
        retrieval_records["settings"].pop("kg")
        retrieval_records.pop("used_kg")
        retrieval_records.pop("reasoning_response")
        answering_records = {
            **retrieval_records,
            "duration": str(end_time - start_time),
            "query": query,
            "prediction": response,
        }
        return answering_records


if __name__ == "__main__":
    pass
