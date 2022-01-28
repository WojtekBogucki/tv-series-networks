import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx import community
import matplotlib.pyplot as plt
import community as community_louvain
import igraph


def get_character_stats(G):
    nodes = list(G.nodes())
    measures = ["degree",
                "weighted_degree",
                "betweenness",
                "eigenvector",
                "closeness",
                "load",
                "pagerank"]
    columns = [measures[0]]
    weight_types = ["line", "scene", "word"]
    for measure in measures[1:]:
        columns = columns + [f"{measure}_{weight_type}" for weight_type in weight_types]
    stats = pd.DataFrame(index=nodes)
    stats[columns[0]] = pd.Series(nx.degree_centrality(G))
    for i, w_type in enumerate(weight_types):   # weighted_degree
        deg_weight = dict(G.degree(weight=f"{w_type}_count"))
        max_deg_weight = max(deg_weight.items(), key=lambda item: item[1])[1]
        dg_weight_cent = {k: np.round(v / max_deg_weight, 3) for k, v in deg_weight.items()}
        stats[columns[1+i]] = pd.Series(dg_weight_cent)
    for i, w_type in enumerate(weight_types):   # betweenness
        distance_dict = {(e1, e2): 1 / weight for e1, e2, weight in G.edges(data=f"{w_type}_count")}
        nx.set_edge_attributes(G, distance_dict, f"{w_type}_dist")
        betweenness_dict = nx.betweenness_centrality(G, weight=f"{w_type}_dist")
        stats[columns[4 + i]] = pd.Series(betweenness_dict)
    for i, w_type in enumerate(weight_types):   # eigenvector
        eigenvector_dict = nx.eigenvector_centrality(G, weight=f"{w_type}_count")
        stats[columns[7 + i]] = pd.Series(eigenvector_dict)
    for i, w_type in enumerate(weight_types):   # closeness
        clossenes_dict = nx.closeness_centrality(G, distance=f"{w_type}_dist")
        max_closeness = max(clossenes_dict.items(), key=lambda item: item[1])[1]
        clossenes_dict = {k: np.round(v / max_closeness, 3) for k, v in clossenes_dict.items()}
        stats[columns[10 + i]] = pd.Series(clossenes_dict)
    for i, w_type in enumerate(weight_types):   # load
        load_dict = nx.load_centrality(G, weight=f"{w_type}_dist")
        stats[columns[13 + i]] = pd.Series(load_dict)
    for i, w_type in enumerate(weight_types):   # pagerank
        page_rank = nx.pagerank(G, weight=f"{w_type}_count")
        stats[columns[16 + i]] = pd.Series(page_rank)
    return stats


def draw_interaction_network_communities(G, weight=None, filename=None, resolution=1.0, method="modularity"):
    '''
    Function that draws an interaction network from given graph
    :param G: networkx graph
    :param weight: name of edge attribute describing edge weight
    :param filename: name of the PNG file to which save the graph
    :param resolution: community detection parameter
    :param method: method of community detection (default greedy_modularity_communities)
    :return:
    '''
    nodes = list(G.nodes())
    edges = G.edges()
    if method == "modularity":
        communities = community.greedy_modularity_communities(G, weight=weight, resolution=resolution)
        print("Found communities:", len(communities))
        com_dict = {character: i for i, com in enumerate(communities) for character in com}
    elif method == "louvain":
        com_dict = community_louvain.best_partition(G, weight=weight, resolution=resolution)
    else:
        print("No community detection selected")
        com_dict = {n: 0 for n in nodes}
    colors = [com_dict[c] for c in nodes]
    if weight == "lines":
        degrees_weight = np.array([v for _, v in G.degree(weight="line_count")])
        edge_width = np.array([G[u][v]['line_count'] for u, v in edges])
        edge_width = edge_width / np.max(edge_width) * 6
    elif weight == "scenes":
        degrees_weight = np.array([v for _, v in G.degree(weight="scene_count")])
        edge_width = np.array([G[u][v]['scene_count'] for u, v in edges])
        edge_width = edge_width / np.max(edge_width) * 6
    elif weight == "words":
        degrees_weight = np.array([v for _, v in G.degree(weight="word_count")])
        edge_width = np.array([G[u][v]['word_count'] for u, v in edges])
        edge_width = edge_width / np.max(edge_width) * 8
    else:
        degrees_weight = np.array([v for _, v in G.degree()])
        edge_width = np.ones(len(edges))
    degrees_weight = degrees_weight/np.max(degrees_weight) * 5000
    pos = nx.spring_layout(G, weight=weight)
    plt.figure(figsize=(16, 9))
    nx.draw_networkx_nodes(G, pos, node_size=degrees_weight, node_color=colors, cmap=plt.get_cmap("Set1"))
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.axis('off')
    # nx.draw_spring(G, with_labels=True, nodelist=nodes, node_size=degrees_weight, width=edge_width, node_color=colors, cmap=plt.get_cmap("Set1"))
    if filename:
        plt.savefig("../figures/{}.png".format(filename))
    else:
        plt.show()