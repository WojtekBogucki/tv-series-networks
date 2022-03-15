import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx import community
import matplotlib.pyplot as plt
import community as community_louvain
import igraph as ig


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
    for i, w_type in enumerate(weight_types):  # weighted_degree
        deg_weight = dict(G.degree(weight=f"{w_type}_count"))
        max_deg_weight = max(deg_weight.items(), key=lambda item: item[1])[1]
        dg_weight_cent = {k: np.round(v / max_deg_weight, 3) for k, v in deg_weight.items()}
        stats[columns[1 + i]] = pd.Series(dg_weight_cent)
    for i, w_type in enumerate(weight_types):  # betweenness
        distance_dict = {(e1, e2): 1 / weight for e1, e2, weight in G.edges(data=f"{w_type}_count")}
        nx.set_edge_attributes(G, distance_dict, f"{w_type}_dist")
        betweenness_dict = nx.betweenness_centrality(G, weight=f"{w_type}_dist")
        stats[columns[4 + i]] = pd.Series(betweenness_dict)
    for i, w_type in enumerate(weight_types):  # eigenvector
        eigenvector_dict = nx.eigenvector_centrality(G, weight=f"{w_type}_count")
        stats[columns[7 + i]] = pd.Series(eigenvector_dict)
    for i, w_type in enumerate(weight_types):  # closeness
        clossenes_dict = nx.closeness_centrality(G, distance=f"{w_type}_dist")
        max_closeness = max(clossenes_dict.items(), key=lambda item: item[1])[1]
        clossenes_dict = {k: np.round(v / max_closeness, 3) for k, v in clossenes_dict.items()}
        stats[columns[10 + i]] = pd.Series(clossenes_dict)
    for i, w_type in enumerate(weight_types):  # load
        load_dict = nx.load_centrality(G, weight=f"{w_type}_dist")
        stats[columns[13 + i]] = pd.Series(load_dict)
    for i, w_type in enumerate(weight_types):  # pagerank
        page_rank = nx.pagerank(G, weight=f"{w_type}_count")
        stats[columns[16 + i]] = pd.Series(page_rank)
    return stats


def draw_character_stats(G: pd.DataFrame, colname: str, filename=None):
    G.loc[:, colname].sort_values(ascending=True).plot(kind="barh")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if filename:
        plt.savefig(f"../figures/{filename}_{colname}.png")
        plt.close()
    else:
        plt.show()


def get_season_networks(path):
    net_seasons = []
    num_seasons = len(
        [f for f in os.listdir(path) if f.startswith("edges_weighted_S") and os.path.isfile(os.path.join(path, f))])
    for i in range(num_seasons):
        edges_weighted_season = pd.read_csv(f"{path}edges_weighted_S{i + 1}.csv")
        net_seasons.append(nx.from_pandas_edgelist(edges_weighted_season,
                                                   source="speaker1",
                                                   target="speaker2",
                                                   edge_attr=["line_count", "scene_count", "word_count"]))
    return net_seasons


def get_episode_networks(path):
    net_episodes = []
    num_seasons = len(
        [f for f in os.listdir(path) if f.startswith("edges_weighted_S") and os.path.isfile(os.path.join(path, f))])
    for i in range(num_seasons):
        season_path = f"{path}season{i + 1}"
        eps_path = os.listdir(season_path)
        print("Season ", i + 1)
        for ep in eps_path:
            edges_weighted_episode = pd.read_csv(os.path.join(season_path, ep))
            net_episodes.append(nx.from_pandas_edgelist(edges_weighted_episode,
                                                        source="speaker1",
                                                        target="speaker2",
                                                        edge_attr=["line_count", "scene_count", "word_count"]))
    return net_episodes


def get_network_stats_by_season(net_seasons):
    seasons = [f"{i + 1}" for i in range(len(net_seasons))]
    columns = ["nodes", "edges", "density", "diameter", "assortativity", "avg_clustering", "avg_shortest_path",
               "transitivity"]
    measures = np.array([[nx.number_of_nodes(net) for net in net_seasons],
                         [nx.number_of_edges(net) for net in net_seasons],
                         [nx.density(net) for net in net_seasons],
                         [nx.diameter(net) for net in net_seasons],
                         [nx.degree_assortativity_coefficient(net, weight="line_count") for net in net_seasons],
                         [nx.average_clustering(net) for net in net_seasons],
                         [nx.average_shortest_path_length(net, weight="line_count") for net in net_seasons],
                         [nx.transitivity(net) for net in net_seasons]]).transpose()
    stats = pd.DataFrame(measures, index=seasons, columns=columns)
    return stats


def get_network_stats_by_episode(net_episodes, episode_dict):
    episodes = episode_dict.keys()
    columns = ["nodes", "edges", "density", "diameter", "assortativity", "avg_clustering", "avg_shortest_path",
               "transitivity"]
    measures = np.array([[nx.number_of_nodes(net) for net in net_episodes],
                         [nx.number_of_edges(net) for net in net_episodes],
                         [nx.density(net) for net in net_episodes],
                         [nx.diameter(net) if nx.is_connected(net) else np.nan for net in net_episodes],
                         [nx.degree_assortativity_coefficient(net, weight="line_count") for net in net_episodes],
                         [nx.average_clustering(net) for net in net_episodes],
                         [nx.average_shortest_path_length(net, weight="line_count") if nx.is_connected(net) else np.nan for net in net_episodes],
                         [nx.transitivity(net) for net in net_episodes]]).transpose()
    stats = pd.DataFrame(measures, index=episodes, columns=columns)
    return stats


def get_network_stats(net):
    columns = ["nodes", "edges", "density", "diameter", "assortativity", "avg_clustering", "avg_shortest_path",
               "transitivity"]
    measures = np.array([nx.number_of_nodes(net),
                         nx.number_of_edges(net),
                         nx.density(net),
                         nx.diameter(net),
                         nx.degree_assortativity_coefficient(net, weight="line_count"),
                         nx.average_clustering(net),
                         nx.average_shortest_path_length(net, weight="line_count"),
                         nx.transitivity(net)])
    stats = {col: measure for col, measure in zip(columns, measures)}
    return stats


def draw_interaction_network_communities(G, weight=None, filename=None, resolution=1.0, method="GM"):
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
    if method is not None:
        method = method.upper()
    if method == "GM":
        communities = community.greedy_modularity_communities(G, weight=weight, resolution=resolution)
        com_dict = {character: i for i, com in enumerate(communities) for character in com}
        colors = [com_dict[c] for c in nodes]
    elif method == "LV":
        com_dict = community_louvain.best_partition(G, weight=weight, resolution=resolution)
        colors = [com_dict[c] for c in nodes]
    elif method in ["SG", "FG", "IM", "LE", "LP", "ML", "WT", "LD"]:
        G_ig = ig.Graph.from_networkx(G)
        if method == "SG":
            colors = G_ig.community_spinglass(weights=weight).membership
        elif method == "FG":
            colors = G_ig.community_fastgreedy(weights=weight).as_clustering().membership
        elif method == "IM":
            colors = G_ig.community_infomap(edge_weights=weight).membership
        elif method == "LE":
            colors = G_ig.community_leading_eigenvector(weights=weight).membership
        elif method == "LP":
            colors = G_ig.community_label_propagation(weights=weight).membership
        elif method == "ML":
            colors = G_ig.community_multilevel(weights=weight).membership
        elif method == "WT":
            colors = G_ig.community_walktrap(weights=weight).as_clustering().membership
        else:
            colors = G_ig.community_leiden(weights=weight).membership
    else:
        print("No community detection selected")
        colors = np.zeros(len(nodes))
    if weight is not None:
        degrees_weight = np.array([v for _, v in G.degree(weight=weight)])
        edge_width = np.array([G[u][v][weight] for u, v in edges])
        edge_width = edge_width / np.max(edge_width) * 8
    else:
        degrees_weight = np.array([v for _, v in G.degree()])
        edge_width = np.ones(len(edges))
    degrees_weight = degrees_weight / np.max(degrees_weight) * 4500
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 16))
    nx.draw_networkx_nodes(G, pos, node_size=degrees_weight, node_color=colors, cmap=plt.get_cmap("Set1"), alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.axis('off')
    # nx.draw_spring(G, with_labels=True, nodelist=nodes, node_size=degrees_weight, width=edge_width, node_color=colors, cmap=plt.get_cmap("Set1"))
    if filename:
        plt.savefig(f"../figures/{filename}.png")
        plt.close()
    else:
        plt.show()
