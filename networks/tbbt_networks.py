from networks.utils import *

# load data
edges_weighted = pd.read_csv("../data/tbbt/edges_weighted.csv")
edges_weighted_top30 = pd.read_csv("../data/tbbt/edges_weighted_top30.csv")
edges_weighted.head()

# create a network
tbbt_net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                                   edge_attr=["line_count", "scene_count", "word_count"])
tbbt_net_top30 = nx.from_pandas_edgelist(edges_weighted_top30, source="speaker1", target="speaker2",
                                         edge_attr=["line_count", "scene_count", "word_count"])

draw_interaction_network_communities(tbbt_net, "line_count", filename="tbbt_lines", method=None)
draw_interaction_network_communities(tbbt_net, "scene_count", filename="tbbt_scenes", method=None)
draw_interaction_network_communities(tbbt_net, "word_count", filename="tbbt_words", method=None)

draw_interaction_network_communities(tbbt_net_top30, "line_count", filename="tbbt_top30_lines", method=None)
draw_interaction_network_communities(tbbt_net_top30, "scene_count", filename="tbbt_top30_scenes", method=None)
draw_interaction_network_communities(tbbt_net_top30, "word_count", filename="tbbt_top30_words", method=None)

# stats
tbbt_char_stats = get_character_stats(tbbt_net)

tbbt_char_stats.loc[:,["pagerank_word", "pagerank_scene"]].sort_values("pagerank_word", ascending=True).plot(kind="barh")
plt.tight_layout()
plt.show()


# by seasons
tbbt_net_seasons = get_season_networks("../data/tbbt/")
tbbt_season_stats = get_network_stats_by_season(tbbt_net_seasons)

tbbt_season_stats["edges"].plot(kind="bar")

draw_interaction_network_communities(tbbt_net_seasons[0], "line_count", method="ML")
draw_interaction_network_communities(tbbt_net_seasons[1], "line_count", method="GM")

# by episodes
show_name = "tbbt"
tbbt_net_episodes = get_episode_networks("../data/tbbt/")

episode_dict = get_episode_dict("../data/tbbt/tbbt_lines_v2.csv")
episode_stats = get_network_stats_by_episode(tbbt_net_episodes, episode_dict, show_name)

draw_interaction_network_communities(tbbt_net_episodes[episode_dict["s09e09"]], "scene_count", method="ML")

episode_stats.plot(kind="scatter", x="transitivity", y="assortativity")
# save networks of all episodes
plt.ioff()
for k, v in episode_dict.items():
    draw_interaction_network_communities(tbbt_net_episodes[v], "line_count", method="ML", filename=f"tbbt/{k}_line_ML")