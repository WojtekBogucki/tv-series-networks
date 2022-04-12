from networks.utils import *
import os

show_name = "the_office"
# load data
edges_weighted = pd.read_csv(f"../data/{show_name}/edges_weighted.csv")
edges_weighted_top30 = pd.read_csv(f"../data/{show_name}/edges_weighted_top30.csv")

# create a network
net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                              edge_attr=["line_count", "scene_count", "word_count"])
net_top30 = nx.from_pandas_edgelist(edges_weighted_top30, source="speaker1", target="speaker2",
                                    edge_attr=["line_count", "scene_count", "word_count"])

# save drawn networks
draw_interaction_network_communities(net, "line_count", filename=f"{show_name}/{show_name}_lines", method="")
draw_interaction_network_communities(net, "scene_count", filename=f"{show_name}/{show_name}_scenes", method="")
draw_interaction_network_communities(net, "word_count", filename=f"{show_name}/{show_name}_words", method="")

draw_interaction_network_communities(net_top30, "line_count", filename=f"{show_name}/{show_name}_top30_lines", method=None)
draw_interaction_network_communities(net_top30, "scene_count", filename=f"{show_name}/{show_name}_top30_scenes", method=None)
draw_interaction_network_communities(net_top30, "word_count", filename=f"{show_name}/{show_name}_top30_words", method=None)

# stats
char_stat_dir = f"../figures/{show_name}/character_stats"

save_character_stats(net_top30, char_stat_dir, "top30")
save_character_stats(net, char_stat_dir, "over_100_lines")


# the office by seasons
net_seasons = get_season_networks(f"../data/{show_name}/")

season_stats = get_network_stats_by_season(net_seasons, show_name)

season_stats.plot(kind="scatter", x="weighted_rating", y="number_of_cliques", s="avg_shortest_path")

plot_corr_mat(season_stats, f"{show_name}/season_corr")

# network stats by season
# os.mkdir(f"../figures/{show_name}/stats_by_season")
for colname in season_stats.columns:
    season_stats[colname].plot(kind="bar", xlabel="Season", ylabel=colname, rot=0)
    plt.savefig(f"../figures/{show_name}/stats_by_season/{colname}.png")
    plt.close()

# seasonal networks
# os.mkdir(f"../figures/{show_name}/season_networks")
for i, season_net in enumerate(net_seasons):
    draw_interaction_network_communities(season_net, "line_count", method="GM", filename=f"{show_name}/season_networks/season{i+1}_line_GM")


draw_interaction_network_communities(net_seasons[8], "line_count", method="SG")
draw_interaction_network_communities(net_seasons[2], "scene_count", method="")
draw_interaction_network_communities(net_seasons[0], "word_count")

# character stats
season_character_stats = pd.DataFrame()
for i, season_net in enumerate(net_seasons):
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
net_episodes = get_episode_networks(f"../data/{show_name}/")
latest_file = [f for f in os.listdir(f"../data/{show_name}/") if f.startswith(f"{show_name}_lines_v")][-1]
episode_dict = get_episode_dict(f"../data/{show_name}/{latest_file}")
episode_stats = get_network_stats_by_episode(net_episodes, episode_dict, show_name)

# add episode rating and number of votes
episode_stats.plot(kind="scatter", x="avg_rating", y="num_votes")
episode_stats.plot(kind="scatter", x="avg_rating", y="assortativity")
os.makedirs(f"../figures/{show_name}/stats_by_episode")
stat_cols = episode_stats.columns
plt.ioff()
ep_corr = episode_stats.corr()
for i, x in enumerate(stat_cols[:-1]):
    for j, y in enumerate(stat_cols[i+1:]):
        if abs(ep_corr.loc[x, y]) > 0.2:
            episode_stats.plot(kind="scatter", x=x, y=y)
            plt.savefig(f"../figures/{show_name}/stats_by_episode/{x}_{y}.png")

plot_corr_mat(episode_stats, f"{show_name}/episode_corr")
plot_corr_mat(episode_stats, f"{show_name}/episode_corr_kendall", method="kendall")
plot_corr_mat(episode_stats, f"{show_name}/episode_corr_spearman", method="spearman")

episode_stats["avg_rating"].plot(kind="hist")
episode_stats["num_votes"].plot(kind="hist")

# save networks of all episodes
plt.ioff()
for k, v in episode_dict.items():
    draw_interaction_network_communities(net_episodes[v], "line_count", method="ML", filename=f"the_office/{k}_line_ML")

draw_interaction_network_communities(net_episodes[episode_dict["s03e07"]], "line_count", method="SG")
draw_interaction_network_communities(net_episodes[episode_dict["s03e07"]], "line_count", method="FG")
draw_interaction_network_communities(net_episodes[episode_dict["s03e07"]], "line_count", method="IM")
draw_interaction_network_communities(net_episodes[episode_dict["s03e07"]], "line_count", method="LE")
draw_interaction_network_communities(net_episodes[episode_dict["s03e07"]], "line_count", method="LP")
draw_interaction_network_communities(net_episodes[episode_dict["s03e07"]], "line_count", method="ML")
draw_interaction_network_communities(net_episodes[episode_dict["s03e07"]], "line_count", method="WT")
draw_interaction_network_communities(net_episodes[episode_dict["s03e07"]], "line_count", method="LD")

draw_interaction_network_communities(net_episodes[episode_dict["s04e12"]], "scene_count", method="GM")

draw_interaction_network_communities(net_episodes[episode_dict["s03e18"]], "line_count",
                                     filename="office_lines_s03e18")
draw_interaction_network_communities(net_episodes[episode_dict["s03e21"]], "line_count",
                                     filename="office_lines_s03e21")


