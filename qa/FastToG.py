import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging
import re
import random
from datetime import datetime
import networkx as nx
from lm import LLM
from graph import KG, Community
from graph.community_algo.louvain import louvain_algo
from qa.QA import QA
from utils.similarity import get_fuzzy_best_match


class FastToG(QA):
    extract_topic_entities_prompt = """Please retrieve 3 topic entities that contribute to answering the following question, and rate their contribution on a scale from 0 to 1. The scores must sum to 1.
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
Given the context (knowledge triples) and a question Q, you are asked to answer the question in curly brackets like {Answer} if you can or {Unknown} if you can not. 
Tips for solution: 
First, please rewrite the question into different basic questions. 
Second, search the context for the information needed to answer these questions. 
Finally, organize all the information and your knowledge to answer.

context:
1. 
Imperial Japanese Army, allegiance, Emperor of Japan

2. 
Yamaji Motoharu, allegiance, Emperor of Japan

3. 
Yamaji Motoharu, military rank, general

Q: Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire ?

A: 
First, the question can be rewrote as: Viscount Yamaji Motoharu was a general of the early Imperial Japanese Army. Which empire did he belong to? 
Second, Based on the context, Viscount Yamaji Motoharu, who was a general in the early Imperial Japanese Army, belonged to the Empire of Japan.
Third, to my knowledge, Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army, which belonged to the Empire of Japan,  which also confirmed by the context.
To sum up, the answer is {Empire of Japan}.

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

A: 
First, the problem can be rewrote as: which team is owned by Steve Bisciotti ? who is the coach of that team ?
Second, context 1 provides geographical information about Millersville. Context 2 lists some football team of National Football League. Context 3 memtioned that Steve Bisciotti is American football player.
Third, context do not directly reveal the current coach of the team owned by Steve Bisciotti. To my knowledge, Steve Bisciotti is the owner of Baltimore Ravens, and the coach of Baltimore Ravens is John Harbaugh. 
To sum up, the answer is {John Harbaugh}.

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

A: 
First, the problem do not need to rewrote beacause it is clear. To solve the problem, we should compare the age or birthday about John Newcombe and Květa Peschke.
Second, context 2 mentioned that John Newcombe was born in 1944 and context 3 mentioned that Květa Peschke was born in 1975. 
Third, compare the above birthday, John Newcomb is not younger than Květa Peschke.
To sum up, the answer is {false}.

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

A: 
First, the question can be broken down into sub-questions: What team did Mychal George Thompson play for? What is the home venue of the San Antonio Spurs?
Second, context by context: Context 1 mentions that the San Antonio Spurs are occupants of the AT&T Center, which is their home venue. Context 2 incorrectly states that the San Antonio Spurs are the home venue of Alamodome and Fort Worth Convention Center; it should have said they used to play at these venues before moving to AT&T Center. Context 3 says the Fort Worth Convention Center is located in Texas and is associated with the San Antonio Spurs.
Finally, organizing the information and my knowledge:
The San Antonio Spurs currently play their home games at the AT&T Center. The historical venues include Alamodome and the Fort Worth Convention Center. If Mychal George Thompson played with the San Antonio Spurs, he could have played at one of these venues, depending on the period he was associated with the team.
Because the context does not specify when Mychal George Thompson played for the San Antonio Spurs or if he did, we cannot explicitly answer which stadium he played at. However, based on the common knowledge, he did not play for the Spurs.
To sum up, the answer is {Unknown}.

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

A: 
First, the question can be rewrote as: Where is Pennsylvania Convention Center? What is the climate there ?
Second, Context 1 mentions the location of the manufacturer of Pennsylvania Convention Center. Context 2 talk about some design and construction of Wilson Brothers & Company. Context 3 provides location information about the Pennsylvania Convention Center.
Third, the Pennsylvania Convention Center is located in Arch Street, Old City, Philadelphia. We can sure that Philadelphia is the city in the Pennsylvania. To my knowledge, the climate in State Pennsylvania is classified as a humid subtropical climate.
To sum up, the answer is {humid subtropical climate}.

Try to use the given context to answer the question only, don't use external knowledge. If you can not find the answer, please answer {Unknown}.
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
        search_depth: int = 5,
        search_max_hop: int = 3,
        search_decline_rate: float = 0,  # original setting: 0.5
        community_max_size: int = 5,
        community_max_candidate: int = 8,
        llm_verbose: bool = False,
        logger: logging.Logger = logging.getLogger(),
    ):
        self.logger = logger
        self.kg = kg
        self.llm_model = LLM(verbose=llm_verbose, logger=self.logger)
        self.search_width = search_width
        self.search_depth = search_depth
        self.search_max_hop = search_max_hop
        self.search_decline_rate = search_decline_rate
        self.community_max_size = community_max_size
        self.community_max_candidate = community_max_candidate

    def retrieve(self, query: str) -> dict[str, any]:
        # === initial phase ===
        # get start community
        start_node = self._get_start_node(query)
        logger.info(f"start community: {start_node}")

        # get header community for each reasoning chain
        self.logger.info("=== Round 1 ===")
        center_community, header_communities = self._local_community_search(
            query, Community(self.kg, [start_node]), self.search_width, [], [start_node]
        )

        if len(header_communities) == 0:
            retrieval_records = {
                "settings": vars(self),
                "reasoning_chain": [[center_community]],
                "early_stop": True,
                "step_count": 0,
            }
            return retrieval_records

        # add header communities to reasoning chains
        reasoning_chains = [[center_community] for _ in range(len(header_communities))]
        self.logger.info("selected community:")
        for idx, comm in enumerate(header_communities):
            self.logger.info(comm)
            reasoning_chains[idx].append(comm)

        # ask LLM to try to answer the question
        is_sufficient = self._reasoning(query, reasoning_chains)

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
                if len(selected_communities) == 0:
                    # no more communities
                    num_of_failed += 1
                    continue
                else:
                    # add selected communities to reasoning chain
                    selected_community = selected_communities[0]
                    self.logger.info(f"selected community:\n{selected_community}")
                    reasoning_chain.append(selected_community)

            if num_of_failed == len(reasoning_chains):
                # all reasoning chains can not find new community
                break

            # ask LLM to try to answer the question
            is_sufficient = self._reasoning(query, reasoning_chains)

            if step_count == self.search_depth:
                # reach maximum search depth
                break

        # collect information
        retrieval_records = {
            "settings": vars(self),
            "reasoning_chain": reasoning_chains,
            "early_stop": not is_sufficient,
            "step_count": step_count,
        }

        return retrieval_records

    def _get_start_node(self, query: str) -> str:
        # ask LLM to extract topic entities
        response = self.llm_model.chat(
            [
                {"role": "system", "content": FastToG.extract_topic_entities_prompt},
                {"role": "user", "content": f"Q: {query}"},
            ]
        )
        results = self.llm_model.parse_response(
            response,
            r"(?:\{|\*{2})(.*?)\s*\(Score:\s*([0-9.]+)\)[ :]*(?:\}|\*{2})",
            {"entity": str, "score": float},
        )
        best_matches = list(
            get_fuzzy_best_match(results[0]["entity"], self.kg.entities)
        )
        return best_matches[0]

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
            communities = self._community_detection(from_community)

            # 2. modularity-based coarse-pruning
            center_community = Community.get_belong_community(
                communities, from_community.center_node
            )
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

    def _community_detection(self, from_community: Community) -> list[Community]:
        """return local partitioned communities"""
        # 1. get n hops subgraph from starting node
        subgraph = self._get_n_hop_subgraph(from_community)

        # 2. partition subgraph into communities
        partitions = louvain_algo(subgraph, self.community_max_size)
        communities = [Community(self.kg, list(partition)) for partition in partitions]
        return communities

    def _get_n_hop_subgraph(self, from_community: Community) -> KG:
        # get n hop subgraph
        subgraph = nx.ego_graph(
            self.kg, from_community.center_node, self.search_max_hop
        )

        # random samples neighbor nodes at different hops with exponential decay; probability=\ro^{hop_count-1}
        prev_hop_nodes = set(list(subgraph[from_community.center_node].keys()))
        visited_nodes = {from_community.center_node}
        visited_nodes.update(prev_hop_nodes)

        for hop_count in range(2, self.search_max_hop + 1):
            # find all nodes in current layer
            current_hop_nodes = set()
            for prev_node in prev_hop_nodes:
                for neighbor_node in subgraph[prev_node].keys():
                    if neighbor_node not in visited_nodes:
                        current_hop_nodes.add(neighbor_node)

            # random remove nodes in the current layer
            removed_nodes = random.sample(
                list(current_hop_nodes),
                int(
                    len(current_hop_nodes)
                    * (1 - self.search_decline_rate ** (hop_count - 1))
                ),
            )
            current_hop_nodes -= set(removed_nodes)
            subgraph.remove_nodes_from(removed_nodes)

            # update for next layer
            visited_nodes.update(current_hop_nodes)
            prev_hop_nodes = current_hop_nodes

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
            if comm_idx >= len(candidate_communities):
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
            if i > 0:
                edges_between_comms = Community.get_edges_between_comms(
                    comm, reasoning_chain[i - 1]
                )
                edge_text = "\n".join(
                    [
                        f"{edge['subject']}, {edge['label']}, {edge['object']}"
                        for edge in edges_between_comms
                    ]
                )
                comm_text = edge_text + "\n" + comm_text
            chain_text_list.append(comm_text)
        return "\n".join(chain_text_list)

    def _reasoning(self, query: str, reasoning_chains: list[list[Community]]) -> bool:
        # transform reasoning chains into text
        reasoning_text_chains = [
            f"{idx}.\n{self._transform_chain_to_text(chain)}"
            for idx, chain in enumerate(reasoning_chains)
        ]
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
        result = self.llm_model.parse_response(response, r"\{(.*?)\}", {"answer": str})

        # check if the answer is valid
        if len(result) == 0:
            return False

        answer = result[0]["answer"].lower()
        if "unknown" in answer:
            return False
        return True

    def _get_used_center_nodes(
        self, reasoning_chains: list[list[Community]]
    ) -> list[str]:
        used_center_nodes = set()
        for reasoning_chain in reasoning_chains:
            for comm in reasoning_chain:
                used_center_nodes.add(comm.center_node)
        return list(used_center_nodes)

    def answer(self, query: str) -> str:
        start_time = datetime.now()
        retrieval_records = self.retrieve(query)

        # transform reasoning chains into text
        reasoning_chains = retrieval_records["reasoning_chain"]
        reasoning_text_chains = [
            f"{idx}.\n{self._transform_chain_to_text(chain)}"
            for idx, chain in enumerate(reasoning_chains)
        ]
        context = "\n\n".join(reasoning_text_chains)

        # ask LLM to try to answer the question
        system_prompt = FastToG.answer_prompt
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

        answering_records = {
            **retrieval_records,
            "duration": str(end_time - start_time),
            "query": query,
            "prediction": response,
        }
        return answering_records


if __name__ == "__main__":
    # query
    query_id = 2
    query = "where did the ceo of salesforce previously work?"

    # KG
    exp_dir = f"./exp_data/query_{query_id}"
    chunk_info_list_paths = [
        os.path.join(exp_dir, pathname)
        for pathname in os.listdir(exp_dir)
        if pathname.startswith("kg_info__")
    ]
    kg = KG(chunk_info_list_paths)
    print(f"# of entities: {len(kg.entities)}")
    print(f"# of relations: {len(kg.relations)}")

    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler("test.log", mode="w")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # FastToG
    fasttog = FastToG(kg, logger=logger, llm_verbose=True)
    # fasttog = FastToG(kg, logger=logger)
    results = fasttog.answer(query)
    logger.info(results)
    print(results["prediction"])
