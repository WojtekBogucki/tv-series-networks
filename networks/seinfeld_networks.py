import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx import community
import matplotlib.pyplot as plt
import community as community_louvain
import igraph
from networks.utils import *

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

seinfeld_stats = get_character_stats(seinfeld_net)

seinfeld_stats.loc[:,["betweenness_word", "betweenness_scene"]].sort_values("betweenness_word", ascending=True).plot(kind="barh")
plt.tight_layout()
plt.show()

# by seasons
seinfeld_net_seasons = get_season_networks("../data/seinfeld/")

seinfeld_season_stats = get_network_stats_by_season(seinfeld_net_seasons)