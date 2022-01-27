import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx import community
import matplotlib.pyplot as plt
import community as community_louvain
import igraph


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