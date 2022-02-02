import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx import community
import matplotlib.pyplot as plt
import community as community_louvain
import igraph as ig

# load data
office_edges_weighted = pd.read_csv("../data/the_office/the_office_edges_weighted.csv")
office_edges_weighted.head()

# create a network
office_net = nx.from_pandas_edgelist(office_edges_weighted, source="speaker1", target="speaker2",
                                     edge_attr=["line_count", "scene_count", "word_count"])

office_net_ig = ig.Graph.from_networkx(office_net)
office_net_ig.vs["name"] = office_net_ig.vs["_nx_name"]

visual_style = {"vertex_label": office_net_ig.vs["_nx_name"], "vertex_color": "red"}
fig, ax = plt.subplots()
layout = office_net_ig.layout("fr")
ig.plot(office_net_ig, layout=layout, target=ax, **visual_style)
plt.axis("off")
plt.show()

EB_community = office_net_ig.community_edge_betweenness(directed=False, weights="line_count").as_clustering()
SG_community = office_net_ig.community_spinglass(weights="line_count")
FG_community = office_net_ig.community_fastgreedy(weights="line_count").as_clustering()
IM_community = office_net_ig.community_infomap(edge_weights="line_count")
LE_community = office_net_ig.community_leading_eigenvector(weights="line_count")
LP_community = office_net_ig.community_label_propagation(weights="line_count")
ML_community = office_net_ig.community_multilevel(weights="line_count")
WT_community = office_net_ig.community_walktrap(weights="line_count").as_clustering()
LD_community = office_net_ig.community_leiden(weights="line_count")

ig.compare_communities(ML_community, SG_community, method="rand")

list(EB_community)