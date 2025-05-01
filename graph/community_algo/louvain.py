import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional
import random
import networkx as nx


def louvain_algo(
    source_multi_graph: nx.MultiGraph, max_community_size: Optional[int] = None
) -> list[set[str]]:
    """return the communities detected by louvain algorithm (int format Ex. [{'B', 'C', 'A', 'Y', 'D', 'Z', 'E'}, {'F', 'H'}, {'G', 'I'}, {'J', 'M', 'L', 'K', 'N'}, {'S', 'R', 'P', 'O', 'Q'}, {'X', 'V', 'U', 'W', 'T'}])"""
    # convert multigraph to graph
    source_normal_graph = _convert_multigraph(source_multi_graph)

    # use networkx's louvain algorithm if community size is not limited
    if max_community_size is None:
        return nx.community.louvain_communities(source_normal_graph)

    # use custom louvain algorithm (from FastToG) if community size is limited
    if source_normal_graph.number_of_nodes() <= max_community_size:
        return [set(source_normal_graph.nodes())]

    # initialize processed graph with node id as name and member info
    processed_graph = nx.Graph()
    node_to_id = {node: idx for idx, node in enumerate(source_normal_graph.nodes())}
    for node in source_normal_graph.nodes():
        processed_graph.add_node(node_to_id[node], member={node})
    for node1, node2, data in source_normal_graph.edges(data=True):
        processed_graph.add_edge(
            node_to_id[node1], node_to_id[node2], weight=data["weight"]
        )
    old_partitions = [data["member"] for _, data in processed_graph.nodes(data=True)]

    while True:
        # one round
        new_inner_partitions = _modularity_optimization(
            processed_graph, max_community_size
        )
        processed_graph = _community_aggregation(processed_graph, new_inner_partitions)

        # check improvement
        new_partitions = [
            data["member"] for _, data in processed_graph.nodes(data=True)
        ]
        if not _modularity_gain(source_normal_graph, old_partitions, new_partitions):
            break
        old_partitions = new_partitions

    return old_partitions


def _convert_multigraph(source_multi_graph: nx.MultiGraph) -> nx.Graph:
    """multigraph -> graph, with weights as number of edges"""
    graph = nx.Graph()
    graph.add_nodes_from(source_multi_graph.nodes(data=True))
    for u, v, data in source_multi_graph.edges(data=True):
        if graph.has_edge(u, v):
            graph[u][v]["weight"] += 1
        else:
            graph.add_edge(u, v, **data, weight=1)
    return graph


def _modularity_optimization(
    processed_graph: nx.Graph, max_community_size: int
) -> list[set[str]]:
    # initialize communities
    node_to_comm = {idx: idx for idx in processed_graph.nodes()}

    # get degree of each node
    node_to_degree = dict(processed_graph.degree(weight="weight"))

    # get total degrees of each community and graph
    comm_to_degrees = {}
    for current_node, comm in node_to_comm.items():
        comm_to_degrees[comm] = (
            comm_to_degrees.get(comm, 0) + node_to_degree[current_node]
        )
    m = processed_graph.size(weight="weight")

    # shuffle comm to avoid local optima
    random_nodes = list(processed_graph.nodes())
    random.shuffle(random_nodes)

    improvement = True
    while improvement:
        improvement = False
        for current_node in random_nodes:
            current_comm = node_to_comm[current_node]
            node_degree = node_to_degree[current_node]

            # temporarily remove from current community
            comm_to_degrees[current_comm] -= node_degree
            node_to_comm[current_node] = -1

            # compute weights to neighboring communities
            weights_to_neighbor_comm = {}
            for neighbor in processed_graph.neighbors(current_node):
                neighbor_comm = node_to_comm[neighbor]
                if neighbor_comm == -1:
                    continue
                weights_to_neighbor_comm[neighbor_comm] = (
                    weights_to_neighbor_comm.get(neighbor_comm, 0)
                    + processed_graph[current_node][neighbor]["weight"]
                )

            # find best community
            best_comm = current_comm
            best_gain = 0
            for (
                neighbor_comm,
                weight_to_neighbor_comm,
            ) in weights_to_neighbor_comm.items():
                gain = weight_to_neighbor_comm / m - node_degree * comm_to_degrees[
                    neighbor_comm
                ] / (2 * m**2)

                if gain > best_gain:
                    # check if size of community size will over limitation
                    size = 0
                    for node, comm in node_to_comm.items():
                        if node == current_node or comm == neighbor_comm:
                            size += len(processed_graph.nodes[node]["member"])
                    if size > max_community_size:
                        continue

                    # update best community
                    best_comm = neighbor_comm
                    best_gain = gain

            # move node to best community
            node_to_comm[current_node] = best_comm
            comm_to_degrees[best_comm] += node_degree

            if best_comm != current_comm:
                improvement = True

    # rebuild partitions
    comm_to_nodes = {}
    for current_node, comm in node_to_comm.items():
        comm_to_nodes[comm] = comm_to_nodes.get(comm, set())
        comm_to_nodes[comm].add(current_node)
    new_partitions = list(comm_to_nodes.values())

    return new_partitions


def _community_aggregation(
    processed_graph: nx.Graph, partitions: list[set[str]]
) -> list[set[str]]:
    # get node to community mapping
    node_to_comm = {}
    for idx, comm in enumerate(partitions):
        for node in comm:
            node_to_comm[node] = idx

    new_graph = nx.Graph()

    # add member info to new graph
    for idx, new_comm in enumerate(partitions):
        new_graph.add_node(idx, member=set())
        for comm in new_comm:
            new_graph.nodes[idx]["member"].update(processed_graph.nodes[comm]["member"])

    # add nodes and edges to new graph
    for node1, node2, data in processed_graph.edges(data=True):
        comm1 = node_to_comm[node1]
        comm2 = node_to_comm[node2]
        weight = data["weight"]

        if not new_graph.has_edge(comm1, comm2):
            new_graph.add_edge(comm1, comm2, weight=0)
        new_graph[comm1][comm2]["weight"] += weight

    return new_graph


def _modularity_gain(
    source_normal_graph: nx.Graph,
    old_partitions: list[set[str]],
    new_partitions: list[set[str]],
) -> bool:
    old_Q = nx.algorithms.community.modularity(source_normal_graph, old_partitions)
    new_Q = nx.algorithms.community.modularity(source_normal_graph, new_partitions)
    return new_Q > old_Q
