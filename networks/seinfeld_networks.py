from networks.utils import *

# load data
edges_weighted = pd.read_csv("../data/seinfeld/edges_weighted.csv")
edges_weighted_top30 = pd.read_csv("../data/seinfeld/edges_weighted_top30.csv")
edges_weighted.head()

# create a network
seinfeld_net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                                       edge_attr=["line_count", "scene_count", "word_count"])
seinfeld_net_top30 = nx.from_pandas_edgelist(edges_weighted_top30, source="speaker1", target="speaker2",
                                       edge_attr=["line_count", "scene_count", "word_count"])

draw_interaction_network_communities(seinfeld_net, "line_count", filename="seinfeld_lines", method=None)
draw_interaction_network_communities(seinfeld_net, "scene_count", filename="seinfeld_scenes", method=None)
draw_interaction_network_communities(seinfeld_net, "word_count", filename="seinfeld_words", method=None)

draw_interaction_network_communities(seinfeld_net_top30, "line_count", filename="seinfeld_top30_lines", method=None)
draw_interaction_network_communities(seinfeld_net_top30, "scene_count", filename="seinfeld_top30_scenes", method=None)
draw_interaction_network_communities(seinfeld_net_top30, "word_count", filename="seinfeld_top30_words", method=None)

# stats
seinfeld_char_stats = get_character_stats(seinfeld_net)

seinfeld_char_stats.loc[:,["betweenness_word", "betweenness_scene"]].sort_values("betweenness_word", ascending=True).plot(kind="barh")
plt.tight_layout()
plt.show()

# by seasons
seinfeld_net_seasons = get_season_networks("../data/seinfeld/")

seinfeld_season_stats = get_network_stats_by_season(seinfeld_net_seasons)

seinfeld_season_stats["nodes"].plot(kind="bar")