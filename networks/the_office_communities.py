import os
import random
import pandas as pd
import numpy as np
import networkx as nx
from networkx import community
import matplotlib.pyplot as plt
import igraph as ig
from timeit import timeit, repeat
from utils import draw_interaction_network_communities, get_episode_networks, get_episode_dict

# load data
office_edges_weighted = pd.read_csv("../data/the_office/edges_weighted.csv")
office_edges_weighted.head()

# create a network
office_net = nx.from_pandas_edgelist(office_edges_weighted, source="speaker1", target="speaker2",
                                     edge_attr=["line_count", "scene_count", "word_count"])

GM_communities = community.greedy_modularity_communities(office_net, weight="line_count")
com_dict = {character: i for i, com in enumerate(GM_communities) for character in com}
nx.set_node_attributes(office_net, com_dict, "GM_community")
nx.attribute_mixing_matrix(office_net, attribute="GM_community")
community.partition_quality(office_net, GM_communities)

com1 = set(GM_communities[0])
com1.add("Michael")
com1_subnet = nx.subgraph(office_net, com1)
draw_interaction_network_communities(com1_subnet, weight="line_count", method=None)
k_ext = com1_subnet.degree(weight="line_count")["Michael"]
k_tot = office_net.degree(weight="line_count")["Michael"]
mu = k_ext / k_tot


def mean_mixing_parameter(g: nx.Graph, attribute: str, weight: str = "line_count"):
    nodes = list(g.nodes())
    nodes_data = g.nodes(data=True)
    mius = []
    for n in nodes:
        n_comm = nodes_data[n][attribute]
        subgraph = nx.subgraph(g, [character for character, data in nodes_data if data[attribute] != n_comm] + [n])
        k_ext = subgraph.degree(weight=weight)[n]
        k_tot = g.degree(weight=weight)[n]
        mius.append(k_ext / k_tot)
    if np.sum(mius) > 0:
        return np.mean(mius)
    else:
        return np.nan


def add_community_attribute(G: nx.Graph, method: str, weight: str = "line_count", resolution: float = 1.0):
    nodes = list(G.nodes())
    com_dict = {character: 0 for character in nodes}
    membership = list(np.zeros(len(nodes), dtype="int"))
    if method == "GM":
        communities = community.greedy_modularity_communities(G, weight=weight, resolution=resolution)
        com_dict = {character: i for i, com in enumerate(communities) for character in com}
    elif method == "LV":
        communities = community.louvain_communities(G, weight=weight, resolution=resolution)
        com_dict = {character: i for i, com in enumerate(communities) for character in com}
    elif method in ["SG", "FG", "IM", "LE", "LP", "ML", "WT", "LD"]:
        G_ig = ig.Graph.from_networkx(G)
        if method == "SG" and nx.is_connected(G):
            membership = G_ig.community_spinglass(weights=weight).membership
        elif method == "FG":
            membership = G_ig.community_fastgreedy(weights=weight).as_clustering().membership
        elif method == "IM":
            membership = G_ig.community_infomap(edge_weights=weight).membership
        elif method == "LE":
            try:
                membership = G_ig.community_leading_eigenvector(weights=weight).membership
            except ig._igraph.InternalError:
                print("Leading eigenvector error")
        elif method == "LP":
            membership = G_ig.community_label_propagation(weights=weight).membership
        elif method == "ML":
            membership = G_ig.community_multilevel(weights=weight).membership
        elif method == "WT":
            membership = G_ig.community_walktrap(weights=weight).as_clustering().membership
        elif method == "LD":
            membership = G_ig.community_leiden(weights=weight, n_iterations=-1, objective_function="modularity",
                                               resolution_parameter=resolution).membership
        com_dict = {char: com for char, com in zip(nodes, membership)}
    nx.set_node_attributes(G, com_dict, "community")
    return G


def get_community_detection_mix_par(show_name: str, weight: str = "line_count", methods: list = None) -> pd.DataFrame:
    if methods is None:
        methods = ["GM", "LV", "SG", "FG", "IM", "LE", "LP", "ML", "WT", "LD"]
    net_episodes = get_episode_networks(f"../data/{show_name}/")
    latest_file = [f for f in os.listdir(f"../data/{show_name}/") if f.startswith(f"{show_name}_lines_v")][-1]
    episode_dict = get_episode_dict(f"../data/{show_name}/{latest_file}")
    mod_df = pd.DataFrame(index=list(episode_dict.keys()))
    for method in methods:
        print(method)
        mix_pars = []
        for ep_code, ep_num in episode_dict.items():
            net_episodes[ep_num] = add_community_attribute(net_episodes[ep_num], method=method, weight=weight)
            mix_par = mean_mixing_parameter(net_episodes[ep_num], "community")
            mix_pars.append(mix_par)
        mix_params = pd.Series(mix_pars)
        mix_params.name = method
        mix_params.index = list(episode_dict.keys())
        mod_df = pd.concat([mod_df, mix_params], axis=1)
    return mod_df


office_net = add_community_attribute(office_net, "IM")
mean_mixing_parameter(office_net, "community")

test_mix_df = get_community_detection_mix_par("the_office")
test_mix_df.mean(axis=0, skipna=True).sort_values()

test_mix_df.var(axis=0)
setup = """import random
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from networkx import community
office_edges_weighted = pd.read_csv("../data/the_office/edges_weighted.csv")
office_net = nx.from_pandas_edgelist(office_edges_weighted, source="speaker1", target="speaker2",
                                     edge_attr=["line_count", "scene_count", "word_count"])
office_net_ig = ig.Graph.from_networkx(office_net)
office_net_ig.vs["name"] = office_net_ig.vs["_nx_name"]                                     
                                     
random.seed(123)
np.random.seed(123)
"""
random.seed(123)
np.random.seed(123)
repeat(stmt="""community.greedy_modularity_communities(office_net, weight="line_count")""", setup=setup, number=100,
       repeat=5)
np.mean(
    repeat(stmt="""office_net_ig.community_fastgreedy(weights="line_count").as_clustering()""", setup=setup, number=100,
           repeat=100))

office_net_ig = ig.Graph.from_networkx(office_net)
office_net_ig.vs["name"] = office_net_ig.vs["_nx_name"]

visual_style = {"vertex_label": office_net_ig.vs["_nx_name"], "vertex_color": "red"}
fig, ax = plt.subplots()
layout = office_net_ig.layout("fr")
ig.plot(office_net_ig, layout=layout, target=ax, **visual_style)
plt.axis("off")
plt.show()

EB_community = office_net_ig.community_edge_betweenness(clusters=4, directed=False, weights="line_count").as_clustering()
SG_community = office_net_ig.community_spinglass(weights="line_count")
node_attr = {char: com for char, com in zip(list(office_net.nodes()), SG_community.membership)}
nx.set_node_attributes(office_net, node_attr, "SG_community")
mean_mixing_parameter(office_net, "SG_community")
FG_community = office_net_ig.community_fastgreedy(weights="line_count").as_clustering()
IM_community = office_net_ig.community_infomap(edge_weights="line_count")
LE_community = office_net_ig.community_leading_eigenvector(weights="line_count")
LP_community = office_net_ig.community_label_propagation(weights="line_count")
ML_community = office_net_ig.community_multilevel(weights="line_count")
WT_community = office_net_ig.community_walktrap(weights="line_count").as_clustering()
LD_community = office_net_ig.community_leiden(weights="line_count")

ig.compare_communities(ML_community, SG_community, method="rand")

list(EB_community)


# other communities
import itertools

def most_central_edge(G):
    centrality = nx.edge_betweenness_centrality(G, weight="line_count")
    return max(centrality, key=centrality.get)


communities = community.girvan_newman(office_net, most_valuable_edge=most_central_edge)
k=3
for comm in itertools.islice(communities, k):
    print(tuple(sorted(c) for c in comm))

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
