import os
import json
from typing import Optional, overload
import networkx as nx
import pyvis
from pyvis.network import Network
from jinja2 import Template


class KG(nx.MultiGraph):
    """
    >>> chunk_info_list_paths = ["chunk_info_list.json"]
    >>> kg = KG(chunk_info_list_paths)
    >>> kg.save_graph("graph.html")
    """

    def __init__(self, chunk_info_list_paths: list[str] = []):
        super().__init__()

        for chunk_info_list_path in chunk_info_list_paths:
            if os.path.exists(chunk_info_list_path):
                with open(chunk_info_list_path, "r") as f:
                    chunk_info_list = json.load(f)
                    for chunk_info in chunk_info_list:
                        self._construct_graph(
                            chunk_info["entities"], chunk_info["triples"]
                        )

    def _construct_graph(
        self, entities: dict[str, str], triples: list[dict[str, any]]
    ) -> None:
        for triple in triples:
            entity1 = triple["subject"]
            entity2 = triple["object"]
            relationship = triple["relationship"]

            try:
                if self.has_node(entity1) and entities.get(entity1) is None:
                    pass
                else:
                    self.add_node(
                        entity1,
                        size=20,
                        color=self._get_color(entity1, entities),
                    )
            except Exception:
                print(f"Error adding node: {entity1}")
            try:
                if self.has_node(entity2) and entities.get(entity2) is None:
                    pass
                else:
                    self.add_node(
                        entity2,
                        size=20,
                        color=self._get_color(entity2, entities),
                    )
            except Exception:
                print(f"Error adding node: {entity2}")
            try:
                self.add_edge(
                    entity1,
                    entity2,
                    label=relationship,
                    subject=entity1,
                    object=entity2,
                )
            except Exception:
                print(f"Error adding edge: {triple}")

    def _get_color(self, target_entity: str, entity_label_dict: dict[str, str]) -> str:
        label2color = {
            "person": "crimson",
            "organization": "darkorange",
            "location": "mediumseagreen",
            "event": "gold",
            "date": "deepskyblue",
            "product": "mediumvioletred",
            "law": "slategray",
            "medical": "orchid",
            "scientific_term": "teal",
            "work_of_art": "indigo",
            "language": "chocolate",
            "nationality": "darkgreen",
            "religion": "firebrick",
            "sport": "navy",
            "weapon": "black",
            "food": "saddlebrown",
            "currency": "olive",
            "disease": "tomato",
            "animal": "royalblue",
            "plant": "forestgreen",
        }

        if target_entity in entity_label_dict:
            return label2color[entity_label_dict[target_entity]]
        else:
            return "lightgray"

    @property
    def entities(self) -> dict[str, dict[str, any]]:
        return {node: self.nodes[node] for node in self.nodes}

    @property
    def relations(self) -> dict[tuple[str, str, str], dict[str, any]]:
        return {
            (edge_info["subject"], edge_info["label"], edge_info["object"]): edge_info
            for edge_info in self.edges.values()
        }

    @overload
    def has_edge(self, node1: str, node2: str) -> bool: ...

    @overload
    def has_edge(self, node1: str, node2: str, label: str) -> bool: ...

    def has_edge(
        self,
        node1: str,
        node2: str,
        label: Optional[str] = None,
    ) -> bool:
        if label is None:
            return super().has_edge(node1, node2)
        elif super().has_edge(node1, node2) is False:
            return False
        elif node2 not in self[node1]:
            return False

        relations = [edge["label"] for edge in self[node1][node2].values()]
        return label in relations

    def save_graph(self, path: str) -> None:
        """path: xxx.html"""
        nt = Network()
        template_path = os.path.join(pyvis.__path__[0], "templates", "template.html")
        with open(template_path, "r") as f:
            nt.template = Template(f.read())

        nt.from_nx(self)
        nt.force_atlas_2based()
        nt.show(path)

    def format_as_trees(self, show_attributes: Optional[list[str]] = None) -> str:
        # count the number of times each entity appears as a subject
        subject_entities = [
            attributes["subject"] for _, attributes in self.relations.items()
        ]
        subject_entities_count = {
            entity: subject_entities.count(entity) for entity in subject_entities
        }
        subject_entities = sorted(
            set(subject_entities),
            key=lambda entity: subject_entities_count[entity],
            reverse=True,
        )

        # format each entity as a tree
        formatted_str = ""
        while len(subject_entities) > 0:
            root_node = subject_entities.pop(0)
            subject_tree_str, visited = self._format_as_tree(
                root_node, show_attributes=show_attributes
            )
            for entity in visited:
                if entity in subject_entities:
                    subject_entities.remove(entity)
            formatted_str += subject_tree_str + "\n\n"

        return formatted_str.strip()

    def _format_as_tree(
        self,
        root_node: str,
        prefix: str = "",
        relation_attributes: dict[str, any] = None,
        show_attributes: Optional[list[str]] = None,
        visited: Optional[set[str]] = None,
    ) -> tuple[str, set[str]]:
        formatted_str = ""
        formatted_str += prefix + root_node
        if relation_attributes is not None:
            relation = relation_attributes["label"]
            if show_attributes is None:
                formatted_str += f" ─── ({relation})"
            else:
                formatted_str += f" ─── ({relation} <{', '.join([f'{attr}={relation_attributes[attr]}' for attr in show_attributes if attr in relation_attributes])}>)"

        if visited is None:
            visited = set()
        if root_node in visited:
            return formatted_str, visited
        visited.add(root_node)

        neighbors = [
            (neighbor, edge)
            for neighbor, edges in self[root_node].items()
            for edge in edges.values()
            if edge["subject"] == root_node
        ]

        for neighbor, relation_attributes in neighbors:
            add_prefix = "└── " if neighbor == neighbors[-1][0] else "├── "
            if prefix == "":
                new_prefix = add_prefix
            elif prefix[-4:] == "└── ":
                new_prefix = prefix[:-4] + "    " + add_prefix
            else:
                new_prefix = prefix[:-4] + "|   " + add_prefix

            neighbor_str, neighbor_visited = self._format_as_tree(
                neighbor,
                new_prefix,
                relation_attributes,
                show_attributes,
                visited.copy(),
            )
            formatted_str += "\n" + neighbor_str
            visited.update(neighbor_visited)

        return formatted_str, visited

    def format_as_triples(self, by: str) -> str:
        triples = sorted(
            [triple for triple in self.relations],
            key=lambda triple: self.relations[triple][by],
        )
        formatted_str = "\n".join([", ".join(triple) for triple in triples])
        return formatted_str

    def format_as_paths(self, by: str) -> str:
        # count the number of times each entity appears as a subject
        subject_entities = [
            attributes["subject"] for _, attributes in self.relations.items()
        ]
        subject_entities_count = {
            entity: subject_entities.count(entity) for entity in subject_entities
        }
        subject_entities = sorted(
            set(subject_entities),
            key=lambda entity: subject_entities_count[entity],
            reverse=True,
        )

        # make paths
        all_paths = []
        while len(subject_entities) > 0:
            root_node = subject_entities.pop(0)
            subject_path, visited = self._format_as_paths(root_node, by)
            for entity in visited:
                if entity in subject_entities:
                    subject_entities.remove(entity)
            all_paths.extend(subject_path)

        formatted_str = "\n".join(
            [" -> ".join([", ".join(triple) for triple in path]) for path in all_paths]
        )

        return formatted_str.strip()

    def _format_as_paths(
        self, root_node: str, by: str, current_path=None, visited=None
    ) -> tuple[list[any], set[str]]:
        if current_path is None:
            current_path = []
        if visited is None:
            visited = set()

        if root_node in visited:
            return [current_path], visited
        visited.add(root_node)

        neighbors = [
            (neighbor, edge)
            for neighbor, edges in self[root_node].items()
            for edge in edges.values()
            if edge["subject"] == root_node
        ]
        neighbors.sort(key=lambda x: x[1][by])

        all_paths = []
        if len(neighbors) == 0:
            all_paths.append(current_path)
        else:
            for neighbor, relation_attributes in neighbors:
                extended_path = current_path + [
                    (root_node, relation_attributes["label"], neighbor)
                ]
                sub_paths, sub_visited = self._format_as_paths(
                    neighbor, by, extended_path, visited.copy()
                )
                all_paths.extend(sub_paths)
                visited.update(sub_visited)

        return all_paths, visited


if __name__ == "__main__":
    chunk_info_list_paths = ["chunk_info_list.json"]
    kg = KG(chunk_info_list_paths)
    kg.save_graph("graph.html")
    print(f"# of entities: {len(kg.entities.keys())}")
    print(f"# of relations: {len(kg.relations.keys())}")
