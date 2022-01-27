import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx import community
import matplotlib.pyplot as plt
import community as community_louvain
import igraph
from networks.utils import draw_interaction_network_communities


# load data
office_edges_weighted = pd.read_csv("../data/the_office/the_office_edges_weighted.csv")
office_edges_weighted.head()

# create a network
office_net = nx.from_pandas_edgelist(office_edges_weighted, source="speaker1", target="speaker2",
                                     edge_attr=["line_count", "scene_count", "word_count"])



# save drawn networks
draw_interaction_network_communities(office_net, "lines", filename="the_office_lines", method=None)
draw_interaction_network_communities(office_net, "scenes", filename="the_office_scenes", method=None)
draw_interaction_network_communities(office_net, "words", filename="the_office_words", method=None)

# stats
# density
density = nx.density(office_net)
print("Network density:", density)

nodes = list(office_net.nodes())

stats = pd.DataFrame(index=nodes, columns=["deg_centrality", "weighted_deg_centrality"])

# centrality
deg_weight_line = dict(office_net.degree(weight="line_count"))
max_deg_weight_line = max(deg_weight_line.items(), key=lambda item: item[1])[1]
dg_weighted_centrality = {k: np.round(v/max_deg_weight_line, 3) for k, v in sorted(deg_weight_line.items(), key=lambda item: item[1], reverse=True)}
print("Weighted degree centrality:", dg_weighted_centrality)
for key, val in dg_weighted_centrality.items():
    stats.loc[key, "weighted_deg_centrality"] = val

dg_centrality = nx.degree_centrality(office_net)
dg_centrality = {k: np.round(v, 2) for k, v in sorted(dg_centrality.items(), key=lambda item: item[1], reverse=True)}
print("Degree centrality:", dg_centrality)
for key, val in dg_centrality.items():
    stats.loc[key, "deg_centrality"] = val

office_net_distance_dict = {(e1, e2): 1 / weight for e1, e2, weight in office_net.edges(data="line_count")}
nx.set_edge_attributes(office_net, office_net_distance_dict, "line_distance")
betweenness_dict = nx.betweenness_centrality(office_net, weight="line_distance")
betweenness_dict = {k: np.round(v, 3) for k, v in sorted(betweenness_dict.items(), key=lambda item: item[1], reverse=True)}
print("Betweenness centrality:", betweenness_dict)

eigenvector_dict = nx.eigenvector_centrality(office_net, weight="line_count")
eigenvector_dict = {k: np.round(v, 2) for k, v in sorted(eigenvector_dict.items(), key=lambda item: item[1], reverse=True)}
print("Eigenvector centrality:", eigenvector_dict)

clossenes_dict = nx.closeness_centrality(office_net, distance="line_distance")
clossenes_dict = {k: np.round(v, 2) for k, v in sorted(clossenes_dict.items(), key=lambda item: item[1], reverse=True)}
print("Closseness centrality:", clossenes_dict)

load_dict = nx.load_centrality(office_net, weight="line_distance")
load_dict = {k: np.round(v, 2) for k, v in sorted(load_dict.items(), key=lambda item: item[1], reverse=True)}
print("Load centrality:", load_dict)

page_rank = nx.pagerank(office_net, weight="line_count")
page_rank = {k: np.round(v, 2) for k, v in sorted(page_rank.items(), key=lambda item: item[1], reverse=True)}
print("PageRank:", page_rank)

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
for i in range(1, 10):
    office_edges_weighted_seasons.append(pd.read_csv("../data/the_office/the_office_edges_weighted_S{}.csv".format(i)))

office_edges_weighted_seasons[0].head()
office_net_seasons = []
for office_edges_weighted_season in office_edges_weighted_seasons:
    office_net_seasons.append(nx.from_pandas_edgelist(office_edges_weighted_season,
                                                      source="speaker1",
                                                      target="speaker2",
                                                      edge_attr=["line_count", "scene_count", "word_count"]))

draw_interaction_network_communities(office_net_seasons[2], "lines")
draw_interaction_network_communities(office_net_seasons[2], "scenes", method=None)
draw_interaction_network_communities(office_net_seasons[0], "words")

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
                                                       edge_attr=["line_count", "scene_count", "word_count"]))

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

draw_interaction_network_communities(office_net_episodes[episode_dict["s03e21"]], "lines")

draw_interaction_network_communities(office_net_episodes[episode_dict["s03e18"]], "lines", filename="office_lines_s03e18")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e21"]], "lines", filename="office_lines_s03e21")

# another algorithm
communities = community.girvan_newman(office_net_episodes[episode_dict["s02e08"]], most_valuable_edge=most_central_edge)
node_groups = []
for com in next(communities):
    node_groups.append(list(com))

print(node_groups)

