import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx import community

# load data
office_edges_weighted = pd.read_csv("../data/the_office/the_office_edges_weighted.csv")

office_edges_weighted.head()

office_net = nx.from_pandas_edgelist(office_edges_weighted, source="speaker1", target="speaker2", edge_attr=["line_count", "scene_count"])
nodes = list(office_net.nodes())
edges = office_net.edges()
degrees = [v for _, v in office_net.degree()]
degrees_lines = [v for _, v in office_net.degree(weight="line_count")]
degrees_scenes = [v for _, v in office_net.degree(weight="scene_count")]
line_counts_width = np.array([office_net[u][v]['line_count'] for u, v in edges])/1000
plt.figure(figsize=(15, 15))
nx.draw_spring(office_net, with_labels=True, nodelist=nodes, node_size=[d/5 for d in degrees_lines], width=line_counts_width)
plt.show()

# stats
density = nx.density(office_net)
print("Network density:", density)
triadic_closure = nx.transitivity(office_net)
print("Triadic closure:", triadic_closure)
betweenness_dict = nx.betweenness_centrality(office_net) # Run betweenness centrality
eigenvector_dict = nx.eigenvector_centrality(office_net) # Run eigenvector centrality

communities = community.greedy_modularity_communities(office_net)