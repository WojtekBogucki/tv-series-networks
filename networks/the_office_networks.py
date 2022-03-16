from networks.utils import *
import os

show_name = "the_office"
# load data
office_edges_weighted = pd.read_csv(f"../data/{show_name}/edges_weighted.csv")
office_edges_weighted_top30 = pd.read_csv(f"../data/{show_name}/edges_weighted_top30.csv")

# create a network
office_net = nx.from_pandas_edgelist(office_edges_weighted, source="speaker1", target="speaker2",
                                     edge_attr=["line_count", "scene_count", "word_count"])
office_net_top30 = nx.from_pandas_edgelist(office_edges_weighted_top30, source="speaker1", target="speaker2",
                                     edge_attr=["line_count", "scene_count", "word_count"])

# save drawn networks
draw_interaction_network_communities(office_net, "line_count", filename=f"{show_name}/{show_name}_lines", method=None)
draw_interaction_network_communities(office_net, "scene_count", filename=f"{show_name}/{show_name}_scenes", method=None)
draw_interaction_network_communities(office_net, "word_count", filename=f"{show_name}/{show_name}_words", method=None)

draw_interaction_network_communities(office_net_top30, "line_count", filename=f"{show_name}/{show_name}_top30_lines", method=None)
draw_interaction_network_communities(office_net_top30, "scene_count", filename=f"{show_name}/{show_name}_top30_scenes", method=None)
draw_interaction_network_communities(office_net_top30, "word_count", filename=f"{show_name}/{show_name}_top30_words", method=None)

# stats
os.mkdir(f"../figures/{show_name}/character_stats")

office_stats = get_character_stats(office_net_top30)

for colname in office_stats.columns:
    draw_character_stats(office_stats, colname, filename=f"{show_name}/character_stats/top30")


# communities
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
office_net_seasons = get_season_networks(f"../data/{show_name}/")

office_season_stats = get_network_stats_by_season(office_net_seasons)

# add rating weighted by number of votes
season_ratings = pd.read_csv("../data/imdb/season_ratings.csv")
season_ratings = season_ratings[season_ratings.originalTitle == "The Office"]
weighted_rating = season_ratings["weighted_rating"].tolist()
office_season_stats["weighted_rating"] = weighted_rating

# network stats by season
# os.mkdir(f"../figures/{show_name}/stats_by_season")
for colname in office_season_stats.columns:
    office_season_stats[colname].plot(kind="bar", xlabel="Season", ylabel=colname)
    plt.savefig(f"../figures/{show_name}/stats_by_season/{colname}.png")
    plt.close()

# seasonal networks
# os.mkdir(f"../figures/{show_name}/season_networks")
for i, season_net in enumerate(office_net_seasons):
    draw_interaction_network_communities(season_net, "line_count", method="GM", filename=f"{show_name}/season_networks/season{i+1}_line_GM")


draw_interaction_network_communities(office_net_seasons[8], "line_count", method="SG")
draw_interaction_network_communities(office_net_seasons[2], "scene_count", method=None)
draw_interaction_network_communities(office_net_seasons[0], "word_count")

season_character_stats = pd.DataFrame()
for i, season_net in enumerate(office_net_seasons):
    season_char_stats = get_character_stats(season_net)
    season_char_stats["season"] = i+1
    season_character_stats = pd.concat([season_character_stats, season_char_stats], axis=0)

character_season_count = season_character_stats.groupby(season_character_stats.index)["season"].size().reset_index(name="season_count").sort_values("season_count", ascending=False)
top_characters = character_season_count.loc[character_season_count.season_count>=5, "index"].tolist()
os.mkdir(f"../figures/{show_name}/character_stats_by_season")
for top_character in top_characters:
    season_character_stats.loc[top_character, ["pagerank_line", "season"]].plot(kind="bar", x="season", rot=0, title=f"PageRank by season for {top_character}")
    plt.savefig(f"../figures/{show_name}/character_stats_by_season/pagerank_line_{top_character}.png")
    plt.close()

# the office by episodes
office_net_episodes = get_episode_networks(f"../data/{show_name}/")
episode_dict = get_episode_dict("../data/the_office/the_office_lines_v6.csv")

episode_stats = get_network_stats_by_episode(office_net_episodes, episode_dict)
episode_stats.plot(kind="scatter", x="transitivity", y="assortativity")

episode_stats["transitivity"].plot(kind="hist")

# save networks of all episodes
plt.ioff()
for k, v in episode_dict.items():
    draw_interaction_network_communities(office_net_episodes[v], "line_count", method="ML", filename=f"the_office/{k}_line_ML")

draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="SG")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="FG")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="IM")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="LE")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="LP")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="ML")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="WT")
draw_interaction_network_communities(office_net_episodes[episode_dict["s03e07"]], "line_count", method="LD")

draw_interaction_network_communities(office_net_episodes[episode_dict["s04e12"]], "scene_count", method="GM")

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
