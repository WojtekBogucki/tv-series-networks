import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx import community

# load data
office_edges_weighted = pd.read_csv("../data/the_office/the_office_edges_weighted.csv")

office_edges_weighted.head()
office_net = nx.from_pandas_edgelist(office_edges_weighted, source="speaker1", target="speaker2",
                                     edge_attr=["line_count", "scene_count"])

def draw_interaction_network(G, weight=None):
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
    plt.show()
    return edge_width

deg_test = draw_interaction_network(office_net)

# stats
density = nx.density(office_net)
print("Network density:", density)
triadic_closure = nx.transitivity(office_net)
print("Triadic closure:", triadic_closure)
betweenness_dict = nx.betweenness_centrality(office_net) # Run betweenness centrality
eigenvector_dict = nx.eigenvector_centrality(office_net) # Run eigenvector centrality

communities = community.greedy_modularity_communities(office_net)