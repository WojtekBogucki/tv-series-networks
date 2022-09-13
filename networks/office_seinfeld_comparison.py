from networks.utils import *

# load data
office_edges_weighted_top30 = pd.read_csv("../data/the_office/edges_weighted_top30.csv")
seinfeld_edges_weighted_top30 = pd.read_csv("../data/seinfeld/edges_weighted_top30.csv")

office_net_seasons = get_season_networks("../data/the_office/")
seinfeld_net_seasons = get_season_networks("../data/seinfeld/")


office_net_top30 = nx.from_pandas_edgelist(office_edges_weighted_top30, source="speaker1", target="speaker2",
                                           edge_attr=["line_count", "scene_count", "word_count"])
seinfeld_net_top30 = nx.from_pandas_edgelist(seinfeld_edges_weighted_top30, source="speaker1", target="speaker2",
                                             edge_attr=["line_count", "scene_count", "word_count"])

# stats
office_stats = get_network_stats(office_net_top30)
seinfeld_stats = get_network_stats(seinfeld_net_top30)

office_season_stats = get_network_stats_by_season(office_net_seasons)
seinfeld_season_stats = get_network_stats_by_season(seinfeld_net_seasons)

for stat in seinfeld_season_stats.columns:
    pd.concat((seinfeld_season_stats[stat].rename("Seinfeld"), office_season_stats[stat].rename("The Office")),
              axis=1).plot(kind="bar", xlabel="Season", ylabel=stat)
    plt.xticks(rotation=0)
    plt.savefig(f"../figures/comparison_{stat}.png")


office_char_stats = get_character_stats(office_net_top30)
seinfeld_char_stats = get_character_stats(seinfeld_net_top30)

measures = ["weighted_degree",
                "betweenness",
                "eigenvector",
                "closeness",
                "load",
                "pagerank"]
for stat in measures:
    fig, (ax1,ax2) = plt.subplots(nrows=2, sharex="all")
    office_char_stats.loc[:, [f"{stat}_word", f"{stat}_line"]].sort_values(f"{stat}_word", ascending=True)[20:].plot(kind="barh", ax=ax1)
    seinfeld_char_stats.loc[:, [f"{stat}_word", f"{stat}_line"]].sort_values(f"{stat}_word", ascending=True)[20:].plot(kind="barh", ax=ax2)
    plt.xticks(rotation=45)
    plt.suptitle(f"{stat} comparison")
    plt.tight_layout()
    plt.savefig(f"../figures/character_comparison_{stat}.png")
