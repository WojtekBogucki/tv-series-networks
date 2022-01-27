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
edges_weighted = pd.read_csv("../data/seinfeld/edges_weighted.csv")
edges_weighted.head()

# create a network
seinfeld_net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                                       edge_attr=["line_count", "scene_count", "word_count"])

draw_interaction_network_communities(seinfeld_net, "lines", filename="seinfeld_lines", method=None)
draw_interaction_network_communities(seinfeld_net, "scenes", filename="seinfeld_scenes", method=None)
draw_interaction_network_communities(seinfeld_net, "words", filename="seinfeld_words", method=None)

# stats
# density
density = nx.density(seinfeld_net)
print("Network density:", density)