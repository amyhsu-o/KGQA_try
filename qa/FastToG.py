import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging
import re
from datetime import datetime
from typing import Optional
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
from lm import LLM
from graph import KG, Community
from graph.community_algo.louvain import louvain_algo
from qa.QA import QA
from utils.similarity import get_fuzzy_best_match, get_cosine_similarity_best_match


class FastToG(QA):
    extract_topic_entities_prompt = """Please retrieve {top_k} topic entities that contribute to answering the following question, and rate their contribution on a scale from 0 to 1. The scores must sum to 1.
Q: Name the president of the country whose main spoken language was Brahui in 1980?
A:
1. {{Brahui Language (Score: 0.4)}}: This is the key clue in the question and helps identify the country being referred to.
2. {{Pakistan (Score: 0.3)}}: The country where Brahui is primarily spoken, necessary for determining its president in 1980.
3. {{President of Pakistan (Score: 0.3)}}: Represents the actual target answer type, and is conceptually central to the question.
"""

    community_score_prompt = """
Please score the knowledge triples' contribution to the question on a scale from 0 to 1. The sum of the scores of all triples must be 1.0.

Premise:
Royal Australian Air Force, mascot, kangaroo  
kangaroo, endemic to, Australian continent  
kangaroo, described by source, The New Student's Reference Work  
Australian continent, belong to, Australia  
Carmel Tebbutt, spouse, Anthony Albanese  
Anthony Albanese, position held, Prime Minister of Australia

Question: The majority party now in the country whose symbol is kangaroo is Australian Labor Party, is it true or false?

Choices:
A.  
Carmel Tebbutt, writing language, English  
A recombinant calcitonin receptor..., language of work or name, English  
Eosinophilia..., language of work or name, English  

B.  
Anthony Albanese, work location, Canberra  
2017 Australian constitutional crisis, location, Canberra  
2017 Australian constitutional crisis, instance of, constitutional crisis  

C.  
Australia, nominated by, Prime  
Australia, country, Australia  
Mason, position held, Chief  
Mason, country of citizenship, Australia  
Knox, position held, Chief  
Knox, country of citizenship, Australia  

D.  
station, officially opened by, Anthony  
station, designed by, Weston  
station, adjacent station, High  

E.  
Tebbutt, sex or gender, female  
Streep, sex or gender, female  
Poletti, sex or gender, female  
Bates, sex or gender, female  

F.  
Albanese, member of political party, Australian  
Carroll, member of political party, Australian  
Carroll, instance of, human  
Freelander, member of political party, Australian  
Freelander, instance of, human  
Freelander, member of political party, Australian  

A:
1. {F (Score: 0.7)}: This is the most relevant because it directly confirms Anthony Albanese's membership in the Australian Labor Party, helping verify if it’s currently the majority party.
2. {B (Score: 0.15)}: Slightly relevant, as it situates the Prime Minister in Canberra and links to a constitutional crisis, giving limited indirect context on political dynamics.
3. {C (Score: 0.1)}: Mildly relevant; shows individuals (Mason, Knox) with political roles in Australia, but doesn't clarify party affiliation.
4. {D (Score: 0.03)}: Barely relevant; shows Anthony opened a station, but no political info is provided.
5. {A (Score: 0.01)}: Not relevant; only discusses English as a language.
6. {E (Score: 0.01)}: Not relevant; focuses on gender of unrelated individuals.

Make sure to follow the format of the example.
"""

    community_reasoning_triple_prompt = """
Given a question and the associated retrieved knowledge graphs, you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No). Make sure to include {Yes} or {No} in your answer.

Tips for solution: 
First, please rewrite the question into different basic questions. 
Second, search the context for the information needed to answer these questions. 
Finally, organize all the information and your knowledge to answer {Yes} or {No}.

context:
1. 
Imperial Japanese Army, allegiance, Emperor of Japan

2. 
Yamaji Motoharu, allegiance, Emperor of Japan

3. 
Yamaji Motoharu, military rank, general

Q: Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire ?

A: {Yes}
First, the question can be rewrote as: Viscount Yamaji Motoharu was a general of the early Imperial Japanese Army. Which empire did he belong to? 
Second, Based on the context, Viscount Yamaji Motoharu, who was a general in the early Imperial Japanese Army, belonged to the Empire of Japan.
Third, to my knowledge, Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army, which belonged to the Empire of Japan,  which also confirmed by the context.
To sum up, the answer is Empire of Japan.

context:
1. 
Steve Bisciotti, country of citizenship, United States of America
Steve Bisciotti, educated at, Severn School
Steve Bisciotti, residence, Millersville
Steve Bisciotti, sport, American football

2. 
Dallas Cowboys, instance of, National Football League
New England Patriots, instance of, National Football League
Kansas City Chiefs, instance of, National Football League
Baltimore Ravens, instance of, National Football League

3. 
Steve Bisciotti, sport, American football

Q: Who is the coach of the team owned by the Steve Bisciotti?

A: {Yes}
First, the problem can be rewrote as: which team is owned by Steve Bisciotti ? who is the coach of that team ?
Second, context 1 provides geographical information about Millersville. Context 2 lists some football team of National Football League. Context 3 memtioned that Steve Bisciotti is American football player.
Third, context do not directly reveal the current coach of the team owned by Steve Bisciotti. To my knowledge, Steve Bisciotti is the owner of Baltimore Ravens, and the coach of Baltimore Ravens is John Harbaugh. 
To sum up, the answer is John Harbaugh.

context:
1. 
Květa Peschke, citizen of, Czech

2. 
John Newcombe, date of birth, +1944-05-23T00:00:00Z

3. 
Květa Peschke, date of birth, +1975-07-09T00:00:00Z

4. 
John Newcombe, country of citizenship, Australia

Q: John Newcombe is younger than Květa Peschke, is it true or false? 

A: {Yes} 
First, the problem do not need to rewrote beacause it is clear. To solve the problem, we should compare the age or birthday about John Newcombe and Květa Peschke.
Second, context 2 mentioned that John Newcombe was born in 1944 and context 3 mentioned that Květa Peschke was born in 1975. 
Third, compare the above birthday, John Newcomb is not younger than Květa Peschke.
To sum up, the answer is false.

context:
1. 
San Antonio Spurs, home venue, Alamodome
San Antonio Spurs, home venue, AT&T Center
San Antonio Spurs, home venue, Fort Worth Convention Center

2. 
AT&T Center, occupant, San Antonio Spurs

3. 
Fort Worth Convention Center, located in the administrative territorial entity, Texas
Fort Worth Convention Center, occupant, San Antonio Spurs

Q: At what stadium did Mychal George Thompson play home games with the San Antonio Spurs?

A: {No}
First, the question can be broken down into sub-questions: What team did Mychal George Thompson play for? What is the home venue of the San Antonio Spurs?
Second, context by context: Context 1 mentions that the San Antonio Spurs are occupants of the AT&T Center, which is their home venue. Context 2 incorrectly states that the San Antonio Spurs are the home venue of Alamodome and Fort Worth Convention Center; it should have said they used to play at these venues before moving to AT&T Center. Context 3 says the Fort Worth Convention Center is located in Texas and is associated with the San Antonio Spurs.
Finally, organizing the information and my knowledge:
The San Antonio Spurs currently play their home games at the AT&T Center. The historical venues include Alamodome and the Fort Worth Convention Center. If Mychal George Thompson played with the San Antonio Spurs, he could have played at one of these venues, depending on the period he was associated with the team.
Because the context does not specify when Mychal George Thompson played for the San Antonio Spurs or if he did, we cannot explicitly answer which stadium he played at. However, based on the common knowledge, he did not play for the Spurs.
To sum up, the answer is Unknown.


context:
1. 
Pennsylvania Convention Center, located on street, Arch Street
Wilson Brothers & Company, manufacturer of, Pennsylvania Convention Center
Pennsylvania Convention Center, instance of, convention center

2. 
Wilson Brothers & Company, manufacturer of, Pennsylvania Convention Center
Mauch Chunk, architect, Wilson Brothers & Company
Pennsylvania School for the Deaf, architect, Wilson Brothers & Company

3. 
Arch Street, located in the administrative territorial entity, Old City
Old City, instance of, old town
Old City, located in the administrative territorial entity, Philadelphia
Philadelphia, State of, Pennsylvania
Pennsylvania, climate, humid subtropical climate

4. 
John C. Winston Company, headquarters location, Arch Street
John C. Winston Company, instance of, book publisher
John C. Winston Company, founded by, John Clark Winston
Winston Science Fiction, publisher, John C. Winston Company

Q: What is the climate in the area around the Pennsylvania Convention Center?

A: {Yes}
First, the question can be rewrote as: Where is Pennsylvania Convention Center? What is the climate there ?
Second, Context 1 mentions the location of the manufacturer of Pennsylvania Convention Center. Context 2 talk about some design and construction of Wilson Brothers & Company. Context 3 provides location information about the Pennsylvania Convention Center.
Third, the Pennsylvania Convention Center is located in Arch Street, Old City, Philadelphia. We can sure that Philadelphia is the city in the Pennsylvania. To my knowledge, the climate in State Pennsylvania is classified as a humid subtropical climate.
To sum up, the answer is humid subtropical climate.

Try to use the given context to answer the question only, don't use external knowledge.
"""

    check_reasoning_result_prompt = """
Given a question and a comment about whether the given context can answer the question.
If the comment says the context *can* answer the question, answer {Yes}. 
If the comment says the context *cannot* answer the question, answer {No}.
Please answer {Yes} or {No} only, without explanation.
"""

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
        search_width: int = 3,
        search_depth: int = 10,
        search_max_hop: int = 3,
        search_multi_center: bool = False,
        community_max_size: int = 5,
        community_max_candidate: int = 8,
        update_hist_community_diff: bool = False,
        integrate_reasoning_response: bool = False,
        llm_verbose: bool = False,
        logger: logging.Logger = logging.getLogger(),
    ):
        self.logger = logger
        self.kg = kg
        self.llm_model = LLM(verbose=llm_verbose, logger=self.logger)

        # follow original setting
        self.search_width = search_width
        self.search_depth = search_depth
        self.search_max_hop = search_max_hop
        self.community_max_size = community_max_size
        self.community_max_candidate = community_max_candidate

        """improvement
        `search_multi_center`: when get n hop subgraph in local community search, using multiple center nodes with similar semantics with original center node to extract subgraph
        `update_hist_community_diff`: update last community in the reasoning chain, when local community search using a little bit different nodes to form the center community in next round
        """
        self.search_multi_center = search_multi_center
        if self.search_multi_center:
            self.kg.entities_emb = self.llm_model.embed(list(self.kg.entities.keys()))
        self.update_hist_community_diff = update_hist_community_diff
        self.integrate_reasoning_response = integrate_reasoning_response

    def retrieve(self, query: str, root_path: Optional[str] = None) -> dict[str, any]:
        # === initial phase ===
        self.logger.info("=== Round 1 ===")
        # get header communities
        start_nodes_score = self._get_start_nodes(query)
        self.logger.info(f"header communities: {start_nodes_score}")

        # add header communities to reasoning chains
        reasoning_chains = [list() for _ in range(len(start_nodes_score))]
        for idx, start_node in enumerate(start_nodes_score):
            reasoning_chains[idx].append(Community(self.kg, [start_node]))

        is_sufficient = False
        step_count = 1
        while not is_sufficient:
            step_count += 1
            self.logger.info(f"=== Round {step_count} ===")

            num_of_failed = 0
            used_center_nodes = self._get_used_center_nodes(reasoning_chains)
            for idx, reasoning_chain in enumerate(reasoning_chains):
                self.logger.info(f"--- Chain {idx} ---")

                # get next community
                center_community, selected_communities = self._local_community_search(
                    query, reasoning_chain[-1], 1, reasoning_chain, used_center_nodes
                )

                # run again from header community when nothing retrieved
                if len(selected_communities) == 0:
                    # no more communities
                    center_community, selected_communities = (
                        self._local_community_search(
                            query,
                            reasoning_chain[0],
                            1,
                            reasoning_chain,
                            used_center_nodes,
                        )
                    )
                    if len(selected_communities) == 0:
                        num_of_failed += 1
                        continue
                else:
                    # update differences in the last community
                    if step_count == 2 or self.update_hist_community_diff:
                        # update last community in the reasoning chain
                        last_community = reasoning_chain[-1]
                        all_nodes = set(last_community.nodes())
                        update = False
                        for node in center_community.nodes():
                            if node not in all_nodes:
                                all_nodes.add(node)
                                update = True
                        if update:
                            reasoning_chain[-1] = Community(self.kg, list(all_nodes))

                    # add selected communities to reasoning chain
                    selected_community = selected_communities[0]
                    self.logger.info(f"selected community:\n{selected_community}")
                    reasoning_chain.append(selected_community)

            if num_of_failed == len(reasoning_chains):
                # all reasoning chains can not find new community
                self.logger.info("No more communities")
                break

            # ask LLM to try to answer the question
            is_sufficient, reason = self._reasoning(query, reasoning_chains)

            if step_count == self.search_depth:
                # reach maximum search depth
                self.logger.info("Reach maximum search depth")
                break

        # collect information
        retrieval_records = {
            "settings": vars(self),
            "reasoning_chain": reasoning_chains,
            "reasoning_response": reason if is_sufficient else None,
            "early_stop": not is_sufficient,
            "step_count": step_count,
        }

        # save reasoning graph
        if root_path is not None:
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            self._save_reasoning_graph(reasoning_chains, root_path)

        return retrieval_records

    def _get_start_nodes(self, query: str) -> str:
        # ask LLM to extract topic entities
        response = self.llm_model.chat(
            [
                {
                    "role": "system",
                    "content": FastToG.extract_topic_entities_prompt.format(
                        top_k=self.search_width
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
        results = self._match_score("entity", results, self.kg.entities.keys())

        return results

    def _match_score(
        self,
        label_name: str,
        score_results: list[dict[str, any]],
        match_targets: list[str],
        threshold: float = None,
    ) -> dict[str, float]:
        """

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

    def _local_community_search(
        self,
        query: str,
        from_community: Community,
        top_k: int,
        hist_chain: list[Community],
        used_center_nodes: list[str],
    ) -> tuple[Community, list[Community]]:
        """return top_k selected communities from center community, which have the same center node as from_community"""
        for try_round in range(3):
            # 1. community detection -- find communities around center node
            if self.search_multi_center is False:
                center_nodes = [from_community.center_node]
            else:
                center_nodes = list(
                    get_cosine_similarity_best_match(
                        self.kg.entities_emb[from_community.center_node],
                        self.kg.entities_emb,
                        threshold=0.8,
                        top_n=5,
                    ).keys()
                )

            self.logger.info(f"center nodes: {center_nodes}")
            communities = self._community_detection(center_nodes)

            # 2. modularity-based coarse-pruning
            center_community = Community.get_belong_community(
                communities, from_community.center_node
            )
            if self.search_multi_center:
                center_nodes = set(center_nodes + list(center_community.nodes()))
                center_community = Community(self.kg, center_nodes)

            if len(hist_chain) == 0:
                hist_chain = [center_community]
            neighbor_communities = Community.get_neighbor_communities(
                communities, center_community
            )
            neighbor_communities = [
                comm
                for comm in neighbor_communities
                if comm.center_node not in used_center_nodes
            ]
            self.logger.info(f"center community:\n{center_community}")
            self.logger.info(f"# of neighbor communities: {len(neighbor_communities)}")

            if len(neighbor_communities) == 0:
                # no neighbor communities
                return center_community, []
            elif len(neighbor_communities) <= top_k:
                # no need to prune
                return center_community, neighbor_communities

            if len(neighbor_communities) <= self.community_max_candidate:
                # no need to do coarse-pruning
                candidate_communities = neighbor_communities
            else:
                candidate_communities = self._community_coarse_pruning(
                    neighbor_communities
                )

            # 3. LLMs-based fine-pruning
            selected_communities = self._community_fine_pruning(
                query, candidate_communities, top_k, hist_chain
            )

            if len(selected_communities) > 0:
                # if no communities are selected, then try again
                break
            else:
                self.logger.warning(f"No community selected, try round {try_round}")
        return center_community, selected_communities

    def _community_detection(self, center_nodes: list[str]) -> list[Community]:
        """return local partitioned communities"""
        # 1. get n hops subgraph from starting node
        subgraph = self._get_n_hop_subgraph(center_nodes)

        # 2. partition subgraph into communities
        partitions = louvain_algo(subgraph, self.community_max_size)
        communities = [Community(self.kg, list(partition)) for partition in partitions]
        return communities

    def _get_n_hop_subgraph(self, center_nodes: list[str]) -> KG:
        # get n hop subgraph
        subgraph = nx.compose_all(
            [nx.ego_graph(self.kg, node, self.search_max_hop) for node in center_nodes]
        )
        for node in center_nodes:
            self.logger.info(f"-- center node: {node} ({subgraph.degree(node)}) ")
        return subgraph

    def _community_coarse_pruning(
        self, neighbor_communities: list[Community]
    ) -> list[Community]:
        comm_score = {comm: comm.community_modularity for comm in neighbor_communities}
        neighbor_communities = sorted(
            neighbor_communities, key=lambda comm: comm_score[comm], reverse=True
        )
        return neighbor_communities[: self.community_max_candidate]

    def _community_fine_pruning(
        self,
        query: str,
        candidate_communities: list[Community],
        top_k: int,
        hist_communities: list[Community],
    ) -> list[Community]:
        # prepare history text
        premises = self._transform_chain_to_text(hist_communities)

        # prepare candidate text
        candidates_text = []
        for comm in candidate_communities:
            option_text = comm.format_as_triples()
            edges_between_hist = Community.get_edges_between_comms(
                comm, hist_communities[-1]
            )
            edge_text = "\n".join(
                [
                    f"{edge['subject']}, {edge['label']}, {edge['object']}"
                    for edge in edges_between_hist
                ]
            )
            if len(edge_text) > 0:
                option_text = edge_text + "\n" + option_text

            candidates_text.append(option_text)

        ALPHABETS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        candidates_text = "\n\n".join(
            [f"{ALPHABETS[idx]}.\n{text}" for idx, text in enumerate(candidates_text)]
        )

        # ask LLM to prune
        system_prompt = FastToG.community_score_prompt
        user_prompt = f"""Premise:
{premises}

Question: {query}

{candidates_text}

Your choice:
"""
        response = self.llm_model.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        results = self.llm_model.parse_response(
            response,
            r"(?:\{|\*{2})(.*?)\s*\(Score:\s*([0-9.]+)\)[ :]*(?:\}|\*{2})",
            {"community": str, "score": float},
        )
        community_scores = {}
        for result in results:
            if result["score"] == 0:
                continue
            comm_alphabet = result["community"]
            if len(comm_alphabet) > 1:
                re_result = re.findall(r"[A-Z]", comm_alphabet)
                if len(re_result) != 1 or len(re_result[0]) != 1:
                    continue
                comm_alphabet = re_result[0]
            comm_idx = ord(comm_alphabet) - ord("A")
            if comm_idx < 0 or comm_idx >= len(candidate_communities):
                continue
            community_scores[comm_idx] = max(
                community_scores.get(comm_idx, 0), result["score"]
            )
        selected_communities = [
            candidate_communities[comm_idx]
            for comm_idx in sorted(
                community_scores.keys(),
                key=lambda x: community_scores.get(x),
                reverse=True,
            )
        ][:top_k]

        return selected_communities

    def _transform_chain_to_text(self, reasoning_chain: list[Community]):
        # transform reasoning chain into text
        chain_text_list = []
        for i, comm in enumerate(reasoning_chain):
            comm_text = comm.format_as_triples()
            if i == 0 and len(comm_text) > 0:
                chain_text_list.append(comm_text)
            elif i > 0:
                edges_between_comms = Community.get_edges_between_comms(
                    comm, reasoning_chain[i - 1]
                )
                edge_text = "\n".join(
                    [
                        f"{edge['subject']}, {edge['label']}, {edge['object']}"
                        for edge in edges_between_comms
                    ]
                )

                if len(comm_text) == 0 and len(edge_text) != 0:
                    chain_text_list.append(edge_text)
                elif len(comm_text) != 0 and len(edge_text) == 0:
                    chain_text_list.append(comm_text)
                elif len(comm_text) != 0 and len(edge_text) != 0:
                    chain_text_list.append(edge_text + "\n\n" + comm_text)
        return "\n\n".join(chain_text_list).strip()

    def _reasoning(self, query: str, reasoning_chains: list[list[Community]]) -> bool:
        # transform reasoning chains into text
        reasoning_text_chains = []
        idx = 0
        for chain in reasoning_chains:
            chain_text = self._transform_chain_to_text(chain)
            if len(chain_text) > 0:
                reasoning_text_chains.append(f"{idx}.\n{chain_text}")
        context = "\n\n".join(reasoning_text_chains)

        # ask LLM to try to answer the question
        system_prompt = FastToG.community_reasoning_triple_prompt
        user_prompt = f"""context:
{context}

Q: {query}
A:"""
        response = self.llm_model.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        result = self.llm_model.parse_response(
            response.lower(), r"\{([yes|no]*)\}", {"answer": str}
        )

        # check if the answer is valid
        if len(result) == 0:
            verify_result = self.llm_model.chat(
                [
                    {
                        "role": "system",
                        "content": FastToG.check_reasoning_result_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\nComment:\n{response}\n\nA:",
                    },
                ]
            )
            return "yes" in verify_result.lower(), response
        elif len(result) == 1 and result[0]["answer"] == "yes":
            return True, response
        else:
            return False, response

    def _get_used_center_nodes(
        self, reasoning_chains: list[list[Community]]
    ) -> list[str]:
        used_center_nodes = set()
        for reasoning_chain in reasoning_chains:
            for comm in reasoning_chain:
                used_center_nodes.add(comm.center_node)
        return list(used_center_nodes)

    def _save_reasoning_graph(
        self, reasoning_chains: list[list[Community]], root_path: str
    ) -> None:
        # copy graph
        graph = copy.deepcopy(self.kg)
        for node in graph.nodes():
            del graph.nodes[node]["color"]

        # color map
        max_round = max([len(chain) for chain in reasoning_chains])
        cmap = plt.get_cmap("turbo")
        gradient = np.linspace(0, 1, max_round * 2 - 1)
        colors = [mcolors.to_hex(cmap(val)) for val in gradient]

        # apply color
        for chain in reasoning_chains:
            for idx, comm in enumerate(chain):
                for node1, node2, info in comm.edges(data=True):
                    graph.nodes[node1]["color"] = min(
                        graph.nodes[node1].get("color", max_round * 2), idx * 2
                    )
                    graph.nodes[node2]["color"] = min(
                        graph.nodes[node2].get("color", max_round * 2), idx * 2
                    )
                    for key in graph[node1][node2]:
                        graph.edges[node1, node2, key]["color"] = min(
                            graph.edges[node1, node2, key].get("color", max_round * 2),
                            idx * 2,
                        )
                if idx > 0:
                    for info in Community.get_edges_between_comms(comm, chain[idx - 1]):
                        node1 = info["subject"]
                        node2 = info["object"]
                        if "color" not in graph.nodes[node1]:
                            graph.nodes[node1]["color"] = idx * 2 - 1
                        if "color" not in graph.nodes[node2]:
                            graph.nodes[node2]["color"] = idx * 2 - 1

                        for key in graph[node1][node2]:
                            graph.edges[node1, node2, key]["color"] = min(
                                graph.edges[node1, node2, key].get(
                                    "color", max_round * 2
                                ),
                                idx * 2 - 1,
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

    def answer(self, query: str, root_str: Optional[str] = None) -> str:
        start_time = datetime.now()
        retrieval_records = self.retrieve(query, root_str)

        reasoning_chains = retrieval_records["reasoning_chain"]
        # logging reasoning chain
        self.logger.info("=== Reasoning Chains ===")
        for idx, chain in enumerate(reasoning_chains):
            self.logger.info(f"--- Chain {idx} ---")
            for comm in chain:
                self.logger.info(comm)

        # transform reasoning chains into text
        reasoning_text_chains = [
            f"{idx}.\n{self._transform_chain_to_text(chain)}"
            for idx, chain in enumerate(reasoning_chains)
        ]
        context = "\n\n".join(reasoning_text_chains)

        # ask LLM to answer the question
        system_prompt = FastToG.answer_prompt
        if (
            self.integrate_reasoning_response
            and retrieval_records["reasoning_response"] is not None
        ):
            user_prompt = f"""Knowledge Graph:
{context}

Hint:
{retrieval_records["reasoning_response"]}

Q: {query}
A:"""
        else:
            user_prompt = f"""Knowledge Graph:
{context}

Q: {query}
A:"""
        if not self.llm_model.verbose:
            self.logger.info(user_prompt)
        response = self.llm_model.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        end_time = datetime.now()

        # modify answering records
        retrieval_records["settings"].pop("logger")
        retrieval_records["settings"].pop("kg")
        retrieval_records["settings"].pop("llm_model")
        retrieval_records["reasoning_chain"] = [
            len(retrieval_records["reasoning_chain"][i])
            for i in range(len(retrieval_records["reasoning_chain"]))
        ]
        answering_records = {
            **retrieval_records,
            "duration": str(end_time - start_time),
            "query": query,
            "prediction": response,
        }
        return answering_records


if __name__ == "__main__":
    pass
