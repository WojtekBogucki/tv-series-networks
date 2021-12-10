import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx import community
import matplotlib.pyplot as plt
from matplotlib import cm
import community as community_louvain


# load data
office_edges_weighted = pd.read_csv("../data/the_office/the_office_edges_weighted.csv")

office_edges_weighted.head()
office_net = nx.from_pandas_edgelist(office_edges_weighted, source="speaker1", target="speaker2",
                                     edge_attr=["line_count", "scene_count"])


def draw_interaction_network(G, weight=None, filename=None):
    nodes = list(G.nodes())
    edges = G.edges()
    if weight == "lines":
        degrees_weight = np.array([v for _, v in G.degree(weight="line_count")])
        edge_width = np.array([G[u][v]['line_count'] for u, v in edges])
        edge_width = edge_width / np.max(edge_width) * 6
    elif weight == "scenes":
        degrees_weight = np.array([v for _, v in G.degree(weight="scene_count")])
        edge_width = np.array([G[u][v]['scene_count'] for u, v in edges])
        edge_width = edge_width / np.max(edge_width) * 6
    else:
        degrees_weight = np.array([v for _, v in G.degree()])
        edge_width = np.ones(len(edges))
    degrees_weight = degrees_weight/np.max(degrees_weight) * 5000
    plt.figure(figsize=(15, 15))
    nx.draw_spring(G, with_labels=True, nodelist=nodes, node_size=degrees_weight, width=edge_width)
    if filename:
        plt.savefig("../figures/{}.png".format(filename))
    else:
        plt.show()


def draw_interaction_network_communities(G, weight=None, filename=None, resolution=1.0):
    nodes = list(G.nodes())
    edges = G.edges()
    communities = community.greedy_modularity_communities(G, weight=weight, resolution=resolution)
    print("Found communities:", len(communities))
    com_dict = {}
    for i, com in enumerate(communities):
        for character in com:
            com_dict[character] = i
    colors = [com_dict[c] for c in nodes]
    if weight == "lines":
        degrees_weight = np.array([v for _, v in G.degree(weight="line_count")])
        edge_width = np.array([G[u][v]['line_count'] for u, v in edges])
        edge_width = edge_width / np.max(edge_width) * 6
    elif weight == "scenes":
        degrees_weight = np.array([v for _, v in G.degree(weight="scene_count")])
        edge_width = np.array([G[u][v]['scene_count'] for u, v in edges])
        edge_width = edge_width / np.max(edge_width) * 6
    else:
        degrees_weight = np.array([v for _, v in G.degree()])
        edge_width = np.ones(len(edges))
    degrees_weight = degrees_weight/np.max(degrees_weight) * 5000
    plt.figure(figsize=(15, 15))
    nx.draw_spring(G, with_labels=True, nodelist=nodes, node_size=degrees_weight, width=edge_width, node_color=colors, cmap=plt.get_cmap("Set1"))
    if filename:
        plt.savefig("../figures/{}.png".format(filename))
    else:
        plt.show()


def draw_interaction_network_communities2(G, weight=None, filename=None):
    nodes = list(G.nodes())
    edges = G.edges()
    com_dict = community_louvain.best_partition(G, weight=weight)
    colors = [com_dict[c] for c in nodes]
    if weight == "lines":
        degrees_weight = np.array([v for _, v in G.degree(weight="line_count")])
        edge_width = np.array([G[u][v]['line_count'] for u, v in edges])
        edge_width = edge_width / np.max(edge_width) * 6
    elif weight == "scenes":
        degrees_weight = np.array([v for _, v in G.degree(weight="scene_count")])
        edge_width = np.array([G[u][v]['scene_count'] for u, v in edges])
        edge_width = edge_width / np.max(edge_width) * 6
    else:
        degrees_weight = np.array([v for _, v in G.degree()])
        edge_width = np.ones(len(edges))
    degrees_weight = degrees_weight/np.max(degrees_weight) * 5000
    plt.figure(figsize=(15, 15))
    nx.draw_spring(G, with_labels=True, nodelist=nodes, node_size=degrees_weight, width=edge_width, node_color=colors, cmap=plt.get_cmap("Set1"))
    if filename:
        plt.savefig("../figures/{}.png".format(filename))
    else:
        plt.show()

# save drawn networks
draw_interaction_network(office_net, "lines", filename="the_office_lines")
draw_interaction_network(office_net, "scenes", filename="the_office_scenes")


# stats
# density
density = nx.density(office_net)
print("Network density:", density)

# centrality
deg_weight_line = dict(office_net.degree(weight="line_count"))
max_deg_weight_line = max(deg_weight_line.items(), key=lambda item: item[1])[1]
dg_weighted_centrality = {k: np.round(v/max_deg_weight_line,3) for k, v in sorted(deg_weight_line.items(), key=lambda item: item[1], reverse=True)}
print("Weighted degree centrality:", dg_weighted_centrality)

dg_centrality = nx.degree_centrality(office_net)
dg_centrality = {k: np.round(v, 2) for k, v in sorted(dg_centrality.items(), key=lambda item: item[1], reverse=True)}
print("Degree centrality:", dg_centrality)

betweenness_dict = nx.betweenness_centrality(office_net, weight="line_count")
betweenness_dict = {k: np.round(v, 3) for k, v in sorted(betweenness_dict.items(), key=lambda item: item[1], reverse=True)}
print("Betweenness centrality:", betweenness_dict)

eigenvector_dict = nx.eigenvector_centrality(office_net, weight="line_count")
eigenvector_dict = {k: np.round(v, 2) for k, v in sorted(eigenvector_dict.items(), key=lambda item: item[1], reverse=True)}
print("Eigenvector centrality:", eigenvector_dict)

office_net_distance_dict = {(e1, e2): 1 / weight for e1, e2, weight in office_net.edges(data="line_count")}
nx.set_edge_attributes(office_net, office_net_distance_dict, "line_distance")
clossenes_dict = nx.closeness_centrality(office_net, distance="line_distance")
clossenes_dict = {k: np.round(v, 2) for k, v in sorted(clossenes_dict.items(), key=lambda item: item[1], reverse=True)}
print("Closseness centrality:", clossenes_dict)

load_dict = nx.load_centrality(office_net, weight="line_count")
load_dict = {k: np.round(v, 2) for k, v in sorted(load_dict.items(), key=lambda item: item[1], reverse=True)}
print("Load centrality:", load_dict)

# assortativity
assort_coef = nx.degree_assortativity_coefficient(office_net, weight="line_count")
print("Degree assortativity coefficient:", np.round(assort_coef,3))

# clustering
clust_coef = nx.average_clustering(office_net, weight="line_count")
print("Clustering coefficient:", np.round(clust_coef,3))

triadic_closure = nx.transitivity(office_net)
print("Triadic closure:", np.round(triadic_closure,3))

# communities
communities = community.greedy_modularity_communities(office_net, weight="line_count")
print("Found communities:", len(communities))
for com in communities:
    print(com)


def most_central_edge(G):
    centrality = nx.edge_betweenness_centrality(G, weight="line_count")
    return max(centrality, key=centrality.get)


communities = community.girvan_newman(office_net, most_valuable_edge=most_central_edge)
node_groups = []
for com in next(communities):
    node_groups.append(list(com))

print(node_groups)

color_map = []
for node in office_net:
    if node in node_groups[0]:
        color_map.append('blue')
    else:
        color_map.append('green')
nx.draw(office_net, node_color=color_map, with_labels=True)
plt.show()

# the office by seasons
office_edges_weighted_seasons = []
for i in range(1,10):
    office_edges_weighted_seasons.append(pd.read_csv("../data/the_office/the_office_edges_weighted_S{}.csv".format(i)))

office_edges_weighted_seasons[0].head()
office_net_seasons = []
for office_edges_weighted_season in office_edges_weighted_seasons:
    office_net_seasons.append(nx.from_pandas_edgelist(office_edges_weighted_season,
                                                      source="speaker1",
                                                      target="speaker2",
                                                      edge_attr=["line_count", "scene_count"]))

draw_interaction_network(office_net_seasons[0], "lines")
draw_interaction_network(office_net_seasons[0], "scenes")

communities = community.greedy_modularity_communities(office_net_seasons[8], weight="scene_count")
print("Found communities:", len(communities))
for com in communities:
    print(com)

communities = community.girvan_newman(office_net_seasons[3], most_valuable_edge=most_central_edge)
node_groups = []
for com in next(communities):
    node_groups.append(list(com))

print(node_groups)

# the office by episodes

office_edges_weighted_episodes = []
for i in range(1,10):
    season_path = "../data/the_office/season{}".format(i)
    eps_path = os.listdir(season_path)
    print("Season ",i)
    for ep in eps_path:
        office_edges_weighted_episodes.append(pd.read_csv(os.path.join(season_path, ep)))

office_net_episodes = []
for office_edges_weighted_episode in office_edges_weighted_episodes:
    office_net_episodes.append(nx.from_pandas_edgelist(office_edges_weighted_episode,
                                                       source="speaker1",
                                                       target="speaker2",
                                                       edge_attr=["line_count", "scene_count"]))

office_raw = pd.read_csv("../data/the_office/the_office_lines_v6.csv")
seasons = office_raw.season.unique()
i = 0
episode_dict = {}
for season in seasons:
    office_raw_season = office_raw[office_raw.season == season]
    episodes = office_raw_season.episode.unique()
    for episode in episodes:
        episode_dict["s{0:02d}e{1:02d}".format(season, episode)] = i
        i += 1

draw_interaction_network_communities(office_net_episodes[episode_dict["s06e01"]], "lines", resolution=0.87)

draw_interaction_network_communities2(office_net_episodes[episode_dict["s04e01"]], "lines")
# communities = community.greedy_modularity_communities(office_net_s3ep19, weight="scene_count")
# print("Found communities:", len(communities))
# com_dict = {}
#
# for i, com in enumerate(communities):
#     for character in com:
#         com_dict[character] = i
# colors = [com_dict[c] for c in office_net_s3ep19.nodes()]

