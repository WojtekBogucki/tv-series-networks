from networks.utils import *
import os
import matplotlib

matplotlib.use('Agg')

for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    print(show_name)
    plt.ioff()
    # load data
    edges_weighted = pd.read_csv(f"../data/{show_name}/edges_weighted.csv")
    edges_weighted_top30 = pd.read_csv(f"../data/{show_name}/edges_weighted_top30.csv")

    # create show networks
    net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                                  edge_attr=["line_count", "scene_count", "word_count"])
    net_top30 = nx.from_pandas_edgelist(edges_weighted_top30, source="speaker1", target="speaker2",
                                        edge_attr=["line_count", "scene_count", "word_count"])

    # save drawn networks
    draw_interaction_network_communities(net, "line_count", filename=f"{show_name}/{show_name}_lines", method=None)
    draw_interaction_network_communities(net, "scene_count", filename=f"{show_name}/{show_name}_scenes", method=None)
    draw_interaction_network_communities(net, "word_count", filename=f"{show_name}/{show_name}_words", method=None)

    draw_interaction_network_communities(net_top30, "line_count", filename=f"{show_name}/{show_name}_top30_lines",
                                         method=None)
    draw_interaction_network_communities(net_top30, "scene_count", filename=f"{show_name}/{show_name}_top30_scenes",
                                         method=None)
    draw_interaction_network_communities(net_top30, "word_count", filename=f"{show_name}/{show_name}_top30_words",
                                         method=None)

    # stats
    char_stat_dir = f"../figures/{show_name}/character_stats"
    os.makedirs(char_stat_dir, exist_ok=True)
    save_character_stats(net_top30, char_stat_dir, "top30")
    save_character_stats(net, char_stat_dir, "over_100_lines")

    # by seasons
    net_seasons = get_season_networks(f"../data/{show_name}/")
    season_stats = get_network_stats_by_season(net_seasons, show_name)
    # season_stats.plot(kind="scatter", x="weighted_rating", y="number_of_cliques", s="avg_shortest_path")
    plot_corr_mat(season_stats, f"{show_name}/season_corr")

    # network stats by season
    season_stats_dir = f"../figures/{show_name}/stats_by_season"
    os.makedirs(season_stats_dir, exist_ok=True)
    for colname in season_stats.columns:
        fig, ax = plt.subplots()
        season_stats[colname].plot(kind="bar", xlabel="Season", ylabel=colname, rot=0, ax=ax)
        plt.savefig(os.path.join(season_stats_dir, f"{colname}.png"))
        plt.close(fig)

    # seasonal networks
    os.makedirs(f"../figures/{show_name}/season_networks", exist_ok=True)
    for i, season_net in enumerate(net_seasons):
        draw_interaction_network_communities(season_net, "line_count", method="GM",
                                             filename=f"{show_name}/season_networks/season{i + 1}_line_GM")

    # character stats by season
    season_character_stats = pd.DataFrame()
    for i, season_net in enumerate(net_seasons):
        season_char_stats = get_character_stats(season_net)
        season_char_stats["season"] = i + 1
        season_character_stats = pd.concat([season_character_stats, season_char_stats], axis=0)

    character_season_count = season_character_stats.groupby(season_character_stats.index)["season"].size().reset_index(
        name="season_count").sort_values("season_count", ascending=False)
    top_characters = character_season_count.loc[character_season_count.season_count >= 5, "index"].tolist()
    os.makedirs(f"../figures/{show_name}/character_stats_by_season", exist_ok=True)
    for top_character in top_characters:
        fig, ax = plt.subplots()
        season_character_stats.loc[top_character, ["pagerank_line", "season"]].plot(kind="bar", x="season", rot=0,
                                                                                    title=f"PageRank by season for {top_character}",
                                                                                    ax=ax)
        plt.savefig(f"../figures/{show_name}/character_stats_by_season/pagerank_line_{top_character}.png")
        plt.close(fig)

    # by episodes
    net_episodes = get_episode_networks(f"../data/{show_name}/")
    latest_file = [f for f in os.listdir(f"../data/{show_name}/") if f.startswith(f"{show_name}_lines_v")][-1]
    episode_dict = get_episode_dict(f"../data/{show_name}/{latest_file}")
    episode_stats = get_network_stats_by_episode(net_episodes, episode_dict, show_name)

    os.makedirs(f"../figures/{show_name}/stats_by_episode", exist_ok=True)
    stat_cols = episode_stats.columns
    ep_corr = episode_stats.corr()
    for i, x in enumerate(stat_cols[:-1]):
        for j, y in enumerate(stat_cols[i + 1:]):
            if abs(ep_corr.loc[x, y]) > 0.2:
                fig, ax = plt.subplots()
                episode_stats.plot(kind="scatter", x=x, y=y, ax=ax)
                plt.savefig(f"../figures/{show_name}/stats_by_episode/{x}_{y}.png")
                plt.close(fig)
                # 2d histograms
                mask = (episode_stats[x].notnull()) & (episode_stats[y].notnull())
                fig, ax = plt.subplots()
                h = ax.hist2d(episode_stats.loc[mask, x], episode_stats.loc[mask, y], bins=10)
                plt.xlabel(x)
                plt.ylabel(y)
                cbar = fig.colorbar(h[3], ax=ax)
                cbar.set_label("Number of observations", rotation=270, labelpad=15)
                plt.savefig(f"../figures/{show_name}/stats_by_episode/hist_{x}_{y}.png")
                plt.close(fig)

    # correlations
    plot_corr_mat(episode_stats, f"{show_name}/episode_corr")
    plot_corr_mat(episode_stats, f"{show_name}/episode_corr_kendall", method="kendall")
    plot_corr_mat(episode_stats, f"{show_name}/episode_corr_spearman", method="spearman")

episode_stats = pd.DataFrame()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    net_episodes = get_episode_networks(f"../data/{show_name}/")
    latest_file = [f for f in os.listdir(f"../data/{show_name}/") if f.startswith(f"{show_name}_lines_v")][-1]
    episode_dict = get_episode_dict(f"../data/{show_name}/{latest_file}")
    ep_stats = get_network_stats_by_episode(net_episodes, episode_dict, show_name)
    ep_stats["show"] = show_name
    episode_stats = pd.concat([episode_stats, ep_stats], axis=0)

os.makedirs(f"../figures/comparison", exist_ok=True)

episode_stats.reset_index(drop=True, inplace=True)
plt.ioff()
for col in episode_stats.columns[:-1]:
    xmin = episode_stats[col].min()
    xmax = episode_stats[col].max()
    plt.figure()
    episode_stats[["show", col]].plot(kind="hist", by="show", layout=(1, 4), figsize=(15, 8), sharey=True, sharex=True,
                                      bins=10, range=(xmin, xmax), density=True, legend=False, title=col)
    plt.savefig(f"../figures/comparison/{col}")
    plt.close()

# similarity matrix
create_similarity_matrix(episode_stats)
