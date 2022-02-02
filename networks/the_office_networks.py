from networks.utils import *

# load data
office_edges_weighted = pd.read_csv("../data/the_office/the_office_edges_weighted.csv")
office_edges_weighted_top30 = pd.read_csv("../data/the_office/edges_weighted_top30.csv")

# create a network
office_net = nx.from_pandas_edgelist(office_edges_weighted, source="speaker1", target="speaker2",
                                     edge_attr=["line_count", "scene_count", "word_count"])
office_net_top30 = nx.from_pandas_edgelist(office_edges_weighted_top30, source="speaker1", target="speaker2",
                                     edge_attr=["line_count", "scene_count", "word_count"])

# save drawn networks
draw_interaction_network_communities(office_net, "line_count", filename="the_office_lines", method=None)
draw_interaction_network_communities(office_net, "scene_count", filename="the_office_scenes", method=None)
draw_interaction_network_communities(office_net, "word_count", filename="the_office_words", method=None)

draw_interaction_network_communities(office_net_top30, "line_count", filename="the_office_top30_lines", method=None)
draw_interaction_network_communities(office_net_top30, "scene_count", filename="the_office_top30_scenes", method=None)
draw_interaction_network_communities(office_net_top30, "word_count", filename="the_office_top30_words", method=None)

# stats
# density
density = nx.density(office_net)
print("Network density:", density)

office_stats = get_character_stats(office_net)
office_stats.loc[:, ["pagerank_word", "pagerank_line"]].sort_values("pagerank_word", ascending=True).plot(kind="barh")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


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
office_net_seasons = get_season_networks("../data/the_office/")

office_season_stats = get_network_stats_by_season(office_net_seasons)

draw_interaction_network_communities(office_net_seasons[2], "line_count", method="SG")
draw_interaction_network_communities(office_net_seasons[2], "scene_count", method=None)
draw_interaction_network_communities(office_net_seasons[0], "word_count")

office_season_stats["nodes"].plot(kind="bar")

communities = community.girvan_newman(office_net_seasons[3], most_valuable_edge=most_central_edge)
node_groups = []
for com in next(communities):
    node_groups.append(list(com))

print(node_groups)

# the office by episodes
office_net_episodes = get_episode_networks("../data/the_office/")


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

draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="SG")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="FG")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="IM")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="LE")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="LP")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="ML")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="WT")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="LD")

draw_interaction_network_communities(office_net_episodes[episode_dict["s03e18"]], "line_count",
                                     filename="office_lines_s03e18")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e21"]], "line_count",
                                     filename="office_lines_s03e21")

# another algorithm
communities = community.girvan_newman(office_net_episodes[episode_dict["s02e08"]], most_valuable_edge=most_central_edge)
node_groups = []
for com in next(communities):
    node_groups.append(list(com))

print(node_groups)
