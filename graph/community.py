import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from graph import KG


class Community(KG):
    def __init__(self, from_kg: KG, nodes: list[str] = []):
        super().__init__()
        self.outer_edges = []
        self._extract_subgraph(from_kg, nodes)
        self.community_modularity = self._calculate_community_modularity(from_kg)

    def _extract_subgraph(self, from_kg: KG, nodes: list[str]) -> None:
        for node in nodes:
            self.add_node(node, **from_kg.entities[node])
        for node1, node2, relation_info in from_kg.edges(data=True):
            if node1 not in nodes and node2 not in nodes:
                continue
            elif node1 in nodes and node2 in nodes:
                self.add_edge(
                    relation_info["subject"],
                    relation_info["object"],
                    **relation_info,
                )
            else:
                self.outer_edges.append(relation_info)

    def _calculate_community_modularity(self, from_kg: KG) -> float:
        # calculate the modularity of the community
        m = from_kg.number_of_edges()
        if m == 0:
            raise ValueError("KG is empty, cannot calculate modularity.")

        community_edges = self.number_of_edges()
        tot_sum = 0
        for node in self.nodes():
            tot_sum += from_kg.degree(node)
        return community_edges - tot_sum**2 / (2 * m)

    @property
    def center_node(self) -> str:
        # find the node with maximum degree
        return max(self.nodes(), key=lambda x: self.degree(x))

    def __str__(self):
        return f"""Community({self.center_node}
nodes={self.nodes()}
{self.format_as_paths()}\n)"""

    @staticmethod
    def get_belong_community(communities: list["Community"], node: str) -> "Community":
        for community in communities:
            if node in community.nodes():
                return community
        return None

    @staticmethod
    def get_neighbor_communities(
        communities: list["Community"], center_community: "Community"
    ) -> list["Community"]:
        neighbor_communities = []
        used_communities_center = set()
        for relation_info in center_community.outer_edges:
            if relation_info["subject"] in center_community.nodes():
                neighbor_node = relation_info["object"]
            elif relation_info["object"] in center_community.nodes():
                neighbor_node = relation_info["subject"]

            neighbor_community = Community.get_belong_community(
                communities, neighbor_node
            )
            if (
                neighbor_community is not None
                and neighbor_community.center_node not in used_communities_center
            ):
                neighbor_communities.append(neighbor_community)
                used_communities_center.add(neighbor_community.center_node)
        return neighbor_communities

    @staticmethod
    def get_edges_between_comms(
        comm1: "Community", comm2: "Community"
    ) -> list[dict[str, str]]:
        edges = []
        for edge in comm1.outer_edges:
            if (
                edge["subject"] in comm1.nodes() and edge["object"] in comm1.nodes()
            ) or (edge["subject"] in comm2.nodes() and edge["object"] in comm2.nodes()):
                continue
            if (
                edge["subject"] in comm1.nodes() and edge["object"] in comm2.nodes()
            ) or (edge["subject"] in comm2.nodes() and edge["object"] in comm1.nodes()):
                edges.append(edge)
        return edges


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
    c = Community(
        kg,
        [
            "feloni, richard",
            "marc benioff",
            "germany israel united states czech republic korea netherlands poland",
            "apple inc",
            "fortune’s 'the 50 companies of tomorrow' list",
        ],
    )
    print(c)
    print()
    print(c.outer_edges)
