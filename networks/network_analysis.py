import matplotlib.pyplot as plt

from networks.utils import *
import os
import matplotlib
import numpy as np
import random

matplotlib.use('Agg')

for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    print(show_name)
    plt.ioff()
    # load data
    edges_weighted = pd.read_csv(f"../data/{show_name}/edges_weighted.csv")
    edges_weighted_top30 = pd.read_csv(f"../data/{show_name}/edges_weighted_top30.csv")
    edges_weighted_all = pd.read_csv(f"../data/{show_name}/edges_weighted_all.csv")

    # create show networks
    net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                                  edge_attr=["line_count", "scene_count", "word_count"])
    net_top30 = nx.from_pandas_edgelist(edges_weighted_top30, source="speaker1", target="speaker2",
                                        edge_attr=["line_count", "scene_count", "word_count"])
    net_all = nx.from_pandas_edgelist(edges_weighted_all, source="speaker1", target="speaker2",
                                      edge_attr=["line_count", "scene_count", "word_count"])

    # save drawn networks
    draw_interaction_network_communities(net, "line_count", filename=f"{show_name}/{show_name}_lines", method=None,
                                         seed=777)
    draw_interaction_network_communities(net, "scene_count", filename=f"{show_name}/{show_name}_scenes", method=None,
                                         seed=777)
    draw_interaction_network_communities(net, "word_count", filename=f"{show_name}/{show_name}_words", method=None,
                                         seed=777)

    draw_interaction_network_communities(net_top30, "line_count", filename=f"{show_name}/{show_name}_top30_lines",
                                         method=None, seed=777)
    draw_interaction_network_communities(net_top30, "scene_count", filename=f"{show_name}/{show_name}_top30_scenes",
                                         method=None, seed=777)
    draw_interaction_network_communities(net_top30, "word_count", filename=f"{show_name}/{show_name}_top30_words",
                                         method=None, seed=777)

    # stats
    char_stat_dir = f"../figures/{show_name}/character_stats"
    os.makedirs(char_stat_dir, exist_ok=True)
    save_character_stats(net_top30, char_stat_dir, "top30")
    save_character_stats(net, char_stat_dir, "over_100_lines")
    # save_character_stats(net_all, char_stat_dir, "all")

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
    for method in ["LD", "ML", "FG"]:
        for i, season_net in enumerate(net_seasons):
            random.seed(777)
            np.random.seed(777)
            draw_interaction_network_communities(season_net, "line_count", method=method,
                                                 filename=f"{show_name}/season_networks/season{i + 1}_line_{method}",
                                                 seed=777)

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
    # save episode networks visualizations
    # os.makedirs(f"../figures/{show_name}/episode_networks", exist_ok=True)
    # for method in ["LD", "ML", "FG"]:
    #     for k, v in episode_dict.items():
    #         random.seed(777)
    #         np.random.seed(777)
    #         draw_interaction_network_communities(net_episodes[v], "line_count", method=method,
    #                                              filename=f"{show_name}/episode_networks/{k}_line_{method}", seed=777)

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

show_names = {
    "the_office": "The Office",
    "seinfeld": "Seinfeld",
    "tbbt": "The Big Bang Theory",
    "friends": "Friends"
}

episode_stats.reset_index(drop=True, inplace=True)
plt.ioff()
for col in episode_stats.columns[:-1]:
    xmin = episode_stats[col].min()
    xmax = episode_stats[col].max()
    col_name = " ".join([x for x in col.split("_")])
    fig, axes = plt.subplots(figsize=(15, 8), ncols=4, nrows=1, sharey=True, sharex=True)
    for ax, show in zip(axes, ["the_office", "seinfeld", "tbbt", "friends"]):
        show_filter = episode_stats["show"] == show
        data = episode_stats.loc[show_filter, [col]].dropna()
        data[col].plot(kind="hist", range=(xmin, xmax), ax=ax, bins=min(15, episode_stats[col].nunique()), legend=False, title=show_names[show],
                       fontsize=13, density=False,
                       weights=list(np.ones_like(data.index) / len(data.index)))
        ax.set_ylabel("")
    fig.suptitle(f"Comparison of  TV series by {col_name}", fontsize=15)
    fig.supxlabel(col_name.capitalize())
    fig.supylabel("Frequency (%)")
    plt.savefig(f"../figures/comparison/{col}")
    plt.close(fig)

# similarity matrix
create_similarity_matrix(episode_stats, episode_stats["show"].values.tolist())
create_similarity_matrix(episode_stats[episode_stats["show"] == "the_office"],
                         episode_stats[episode_stats["show"] == "the_office"].index.values.tolist(),
                         filename="the_office/similarity_matrix")
create_similarity_matrix(episode_stats[episode_stats["show"] == "seinfeld"],
                         episode_stats[episode_stats["show"] == "seinfeld"].index.values.tolist(),
                         filename="seinfeld/similarity_matrix")
create_similarity_matrix(episode_stats[episode_stats["show"] == "tbbt"],
                         episode_stats[episode_stats["show"] == "tbbt"].index.values.tolist(),
                         filename="tbbt/similarity_matrix")
create_similarity_matrix(episode_stats[episode_stats["show"] == "friends"],
                         episode_stats[episode_stats["show"] == "friends"].index.values.tolist(),
                         filename="friends/similarity_matrix")

character_stats = pd.DataFrame()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    edges_weighted = pd.read_csv(f"../data/{show_name}/edges_weighted_top30.csv")
    net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                                  edge_attr=["line_count", "scene_count", "word_count"])
    char_stats = get_character_stats(net)
    char_stats["show"] = show_name
    character_stats = pd.concat([character_stats, char_stats], axis=0)

colname = "betweenness_line"
character_stats.index = character_stats.index.str.lower()
top_char_stats = character_stats.loc[character_stats["betweenness_line"] > 0, [colname, "show"]]
top_char_stats.sort_values(by=colname, ascending=True)[colname].transpose().plot(kind="barh",
                                                                                 color=
                                                                                 top_char_stats.sort_values(by=colname,
                                                                                                            ascending=True)[
                                                                                     'show'].map(
                                                                                     {'the_office': plt.cm.Set1(0),
                                                                                      'friends': plt.cm.Set1(3),
                                                                                      'seinfeld': plt.cm.Set1(6),
                                                                                      'tbbt': plt.cm.Set1(8)}))
plt.xticks(rotation=0)
plt.xlabel("Betweenness")
plt.title("Betweenness of characters by line count")
plt.tight_layout()
plt.savefig(f"../figures/comparison/characters_{colname}_top30.png")
plt.close()


# scatterplot
from adjustText import adjust_text

def character_comparison(character_stats, colname, colname2):
    top_char_stats = character_stats.loc[character_stats["betweenness_line"] > 0, [colname, colname2, "show"]]
    fig, ax = plt.subplots(figsize=(20, 10))
    top_char_stats.plot(kind="scatter",
                        x=colname,
                        y=colname2,
                        s=120,
                        linewidth=0,
                        ax=ax,
                        c=top_char_stats['show'].map({'the_office': 0,
                                                      'friends': 1,
                                                      'seinfeld': 2,
                                                      'tbbt': 3}),
                        cmap=plt.get_cmap("Set1"),
                        colorbar=False,
                        grid=True,
                        fontsize=15)
    # for idx, row in top_char_stats.iterrows():
    #     ax.annotate(idx, (row[colname], row[colname2]), xytext=(10, -5), textcoords='offset points', fontsize=15,
    #                 family='sans-serif', color='darkslategrey')
    texts = [plt.text(row[colname], row[colname2], idx, fontsize=20, family='sans-serif', color='darkslategrey') for idx, row in top_char_stats.iterrows()]
    adjust_text(texts)
    plt.title("Comparison of leading characters", fontsize=18)
    plt.xlabel(colname.replace("_", " by "), fontsize=15)
    plt.ylabel(colname2.replace("_", " by "), fontsize=15)
    plt.savefig(f"../figures/comparison/{colname}_{colname2}.png")
    plt.close(fig)


character_comparison(character_stats, "betweenness_line", "pagerank_line")
character_comparison(character_stats, "betweenness_scene", "pagerank_scene")
character_comparison(character_stats, "betweenness_word", "pagerank_word")

# correlations
plot_corr_mat(episode_stats.drop("show", axis=1), f"comparison/episode_corr")
plot_corr_mat(episode_stats[episode_stats.runtime < 30].drop("show", axis=1), f"comparison/episode_corr_below_30_min")