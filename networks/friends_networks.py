from networks.utils import *

# load data
edges_weighted = pd.read_csv("../data/friends/edges_weighted.csv")
edges_weighted_top30 = pd.read_csv("../data/friends/edges_weighted_top30.csv")
edges_weighted.head()

# create a network
friends_net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                                      edge_attr=["line_count", "scene_count", "word_count"])
friends_net_top30 = nx.from_pandas_edgelist(edges_weighted_top30, source="speaker1", target="speaker2",
                                       edge_attr=["line_count", "scene_count", "word_count"])

draw_interaction_network_communities(friends_net, "line_count", filename="friends_lines", method=None)
draw_interaction_network_communities(friends_net, "scene_count", filename="friends_scenes", method=None)
draw_interaction_network_communities(friends_net, "word_count", filename="friends_words", method=None)

draw_interaction_network_communities(friends_net_top30, "line_count", filename="friends_top30_lines", method=None)
draw_interaction_network_communities(friends_net_top30, "scene_count", filename="friends_top30_scenes", method=None)
draw_interaction_network_communities(friends_net_top30, "word_count", filename="friends_top30_words", method=None)


friends_net_seasons = get_season_networks("../data/friends/")

draw_interaction_network_communities(friends_net_seasons[0], "line_count", method="SG")
draw_interaction_network_communities(friends_net_seasons[1], "line_count", method="ML")
draw_interaction_network_communities(friends_net_seasons[2], "line_count", method="ML")

show_name = 'friends'
friends_net_episodes = get_episode_networks("../data/friends/")

episode_dict = get_episode_dict("../data/friends/friends_lines_v2.csv")

episode_stats = get_network_stats_by_episode(friends_net_episodes, episode_dict, show_name)
episode_stats.plot(kind="scatter", x="avg_rating", y="num_votes")
# save networks of all episodes
plt.ioff()
for k, v in episode_dict.items():
    draw_interaction_network_communities(friends_net_episodes[v], "line_count", method="ML", filename=f"friends/{k}_line_ML")

draw_interaction_network_communities(friends_net_episodes[episode_dict["s01e02"]], "line_count", method="ML")