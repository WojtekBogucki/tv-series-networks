import os
import pandas as pd
import numpy as np
import random
import networkx as nx
from networkx import community
import matplotlib.pyplot as plt
import igraph as ig
import seaborn as sns
from scipy.spatial.distance import euclidean, pdist, squareform
from time import time
from tqdm import tqdm


def max_degree(net: nx.Graph, weight: str) -> int:
    return max(net.degree(weight=weight), key=lambda x: x[1])[1]


def get_character_stats(G: nx.Graph) -> pd.DataFrame:
    weight_types = ["line", "scene", "word"]
    nodes = list(G.nodes())
    measures = ["degree",
                "weighted_degree",
                "betweenness",
                "eigenvector",
                "closeness",
                "load",
                "pagerank"]
    columns = [measures[0]]
    for measure in measures[1:]:
        columns = columns + [f"{measure}_{weight_type}" for weight_type in weight_types]
    stats = pd.DataFrame(index=nodes)
    stats[columns[0]] = pd.Series(nx.degree_centrality(G))
    for i, w_type in enumerate(weight_types):  # weighted_degree
        deg_weight = dict(G.degree(weight=f"{w_type}_count"))
        max_deg_weight = max(deg_weight.items(), key=lambda item: item[1])[1]
        dg_weight_cent = {k: np.round(v / max_deg_weight, 3) for k, v in deg_weight.items()}
        stats[columns[1 + i]] = pd.Series(dg_weight_cent)
    for i, w_type in enumerate(weight_types):  # betweenness
        distance_dict = {(e1, e2): 1 / weight for e1, e2, weight in G.edges(data=f"{w_type}_count")}
        nx.set_edge_attributes(G, distance_dict, f"{w_type}_dist")
        betweenness_dict = nx.betweenness_centrality(G, weight=f"{w_type}_dist")
        stats[columns[4 + i]] = pd.Series(betweenness_dict)
    for i, w_type in enumerate(weight_types):  # eigenvector
        eigenvector_dict = nx.eigenvector_centrality(G, weight=f"{w_type}_count")
        stats[columns[7 + i]] = pd.Series(eigenvector_dict)
    for i, w_type in enumerate(weight_types):  # closeness
        clossenes_dict = nx.closeness_centrality(G, distance=f"{w_type}_dist")
        max_closeness = max(clossenes_dict.items(), key=lambda item: item[1])[1]
        clossenes_dict = {k: np.round(v / max_closeness, 3) for k, v in clossenes_dict.items()}
        stats[columns[10 + i]] = pd.Series(clossenes_dict)
    for i, w_type in enumerate(weight_types):  # load
        load_dict = nx.load_centrality(G, weight=f"{w_type}_dist")
        stats[columns[13 + i]] = pd.Series(load_dict)
    for i, w_type in enumerate(weight_types):  # pagerank
        page_rank = nx.pagerank(G, weight=f"{w_type}_count")
        stats[columns[16 + i]] = pd.Series(page_rank)
    return stats


def draw_character_stats(data: pd.DataFrame, colname: str, filename: str = None) -> None:
    data.loc[data[colname] > 0, colname].sort_values(ascending=True).plot(kind="barh")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if filename:
        plt.savefig(f"../figures/{filename}_{colname}.png")
        plt.close()
    else:
        plt.show()


def save_character_stats(G: nx.Graph, path: str, file_prefix: str = "") -> None:
    if not os.path.exists(path):
        os.mkdir(path)
    stats = get_character_stats(G)
    for stat in stats.columns:
        draw_character_stats(stats, stat, filename=os.path.join(path, file_prefix))


def get_season_networks(path: str) -> list[nx.Graph]:
    net_seasons = []
    num_seasons = len(
        [f for f in os.listdir(path) if f.startswith("edges_weighted_S") and os.path.isfile(os.path.join(path, f))])
    for i in range(num_seasons):
        edges_weighted_season = pd.read_csv(f"{path}edges_weighted_S{i + 1}.csv")
        net_seasons.append(nx.from_pandas_edgelist(edges_weighted_season,
                                                   source="speaker1",
                                                   target="speaker2",
                                                   edge_attr=["line_count", "scene_count", "word_count"]))
    return net_seasons


def get_episode_networks(path: str) -> list[nx.Graph]:
    net_episodes = []
    num_seasons = len(
        [f for f in os.listdir(path) if f.startswith("edges_weighted_S") and os.path.isfile(os.path.join(path, f))])
    for i in range(num_seasons):
        season_path = f"{path}season{i + 1}"
        eps_path = os.listdir(season_path)
        print("Season ", i + 1)
        for ep in eps_path:
            edges_weighted_episode = pd.read_csv(os.path.join(season_path, ep))
            net_episodes.append(nx.from_pandas_edgelist(edges_weighted_episode,
                                                        source="speaker1",
                                                        target="speaker2",
                                                        edge_attr=["line_count", "scene_count", "word_count"]))
    return net_episodes


def get_episode_networks_limit(path: str, limit: int) -> list[nx.Graph]:
    net_episodes = []
    num_seasons = len(
        [f for f in os.listdir(path) if f.startswith("edges_weighted_S") and os.path.isfile(os.path.join(path, f))])
    for i in range(num_seasons):
        season_path = f"{path}season{i + 1}"
        eps_path = os.listdir(season_path)
        print("Season ", i + 1)
        for ep in eps_path:
            edges_weighted_episode = pd.read_csv(os.path.join(season_path, ep))
            edges_weighted_episode = edges_weighted_episode[edges_weighted_episode["line_count"] >= limit]
            net_episodes.append(nx.from_pandas_edgelist(edges_weighted_episode,
                                                        source="speaker1",
                                                        target="speaker2",
                                                        edge_attr=["line_count", "scene_count", "word_count"]))
    return net_episodes


def get_network_stats_by_season(net_seasons: list[nx.Graph],
                                show_name: str,
                                weight: str = "line_count",
                                comm_det_method: str = "LD") -> pd.DataFrame:
    seasons = [f"{i + 1}" for i in range(len(net_seasons))]
    columns = ["nodes", "edges", "max_degree", "density", "diameter", "assortativity", "avg_clustering",
               "avg_shortest_path",
               "transitivity", "number_of_cliques", "clique_number", "weighted_rating", "avg_viewership",
               f"number_of_communities_{comm_det_method}", "gini_coef"]
    season_ratings = pd.read_csv("../data/imdb/season_ratings.csv")
    season_ratings = season_ratings[season_ratings.originalTitle == show_name]
    season_view = pd.read_csv("../data/viewership/season_viewership.csv")
    season_view = season_view[season_view.show == show_name]
    measures = np.array([[nx.number_of_nodes(net) for net in net_seasons],
                         [nx.number_of_edges(net) for net in net_seasons],
                         [max_degree(net, weight) for net in net_seasons],
                         [nx.density(net) for net in net_seasons],
                         [nx.diameter(net) for net in net_seasons],
                         [nx.degree_assortativity_coefficient(net, weight=weight) for net in net_seasons],
                         [nx.average_clustering(net) for net in net_seasons],
                         [nx.average_shortest_path_length(net, weight=weight) for net in net_seasons],
                         [nx.transitivity(net) for net in net_seasons],
                         [nx.graph_number_of_cliques(net) for net in net_seasons],
                         [nx.graph_clique_number(net) for net in net_seasons],
                         season_ratings["weighted_rating"].tolist(),
                         season_view["avg_viewership"].tolist(),
                         [len(np.unique(detect_communities(net, method="GM", weight=weight))) for net in net_seasons],
                         [gini_coefficient(np.array(list(dict(net.degree(weight=weight)).values()))) for net in
                          net_seasons]
                         ]).transpose()
    stats = pd.DataFrame(measures, index=seasons, columns=columns)
    return stats


def get_network_stats_by_episode(net_episodes: list[nx.Graph],
                                 episode_dict: dict,
                                 show_name: str,
                                 weight: str = "line_count",
                                 comm_det_method: str = "LD") -> pd.DataFrame:
    episodes = episode_dict.keys()
    columns = ["nodes", "edges", "max_degree", "density", "diameter", "assortativity", "avg_clustering",
               "avg_shortest_path",
               "transitivity", "number_connected_components", "number_of_cliques", "clique_number", "avg_rating",
               "num_votes",
               "runtime", "viewership", f"number_of_communities_{comm_det_method}", "gini_coef"]
    ratings = pd.read_csv("../data/imdb/episode_ratings.csv")
    ratings = ratings[ratings.originalTitle == show_name]
    viewership = pd.read_csv("../data/viewership/viewership.csv")
    viewership = viewership[viewership.show == show_name]
    measures = np.array([[nx.number_of_nodes(net) for net in net_episodes],
                         [nx.number_of_edges(net) for net in net_episodes],
                         [max_degree(net, weight) for net in net_episodes],
                         [nx.density(net) for net in net_episodes],
                         [nx.diameter(net) if nx.is_connected(net) else np.nan for net in net_episodes],
                         [nx.degree_assortativity_coefficient(net, weight=weight) for net in net_episodes],
                         [nx.average_clustering(net) for net in net_episodes],
                         [nx.average_shortest_path_length(net, weight=weight) if nx.is_connected(net) else np.nan for
                          net in net_episodes],
                         [nx.transitivity(net) for net in net_episodes],
                         [nx.number_connected_components(net) for net in net_episodes],
                         [nx.graph_number_of_cliques(net) for net in net_episodes],
                         [nx.graph_clique_number(net) for net in net_episodes],
                         ratings["averageRating"].tolist(),
                         ratings["numVotes"].tolist(),
                         ratings["runtimeMinutes_y"].tolist(),
                         viewership["viewership"].tolist(),
                         [len(np.unique(detect_communities(net, method=comm_det_method, weight=weight))) for net in
                          net_episodes],
                         [gini_coefficient(np.array(list(dict(net.degree(weight=weight)).values()))) for net in
                          net_episodes]
                         ]).transpose()
    stats = pd.DataFrame(measures, index=episodes, columns=columns)
    return stats


def get_network_stats(net: nx.Graph) -> dict:
    columns = ["nodes", "edges", "max_degree", "density", "diameter", "assortativity", "avg_clustering",
               "avg_shortest_path",
               "transitivity"]
    measures = np.array([nx.number_of_nodes(net),
                         nx.number_of_edges(net),
                         max_degree(net, weight="line_count"),
                         nx.density(net),
                         nx.diameter(net),
                         nx.degree_assortativity_coefficient(net, weight="line_count"),
                         nx.average_clustering(net),
                         nx.average_shortest_path_length(net, weight="line_count"),
                         nx.transitivity(net)
                         ])
    stats = {col: measure for col, measure in zip(columns, measures)}
    return stats


def get_movie_network_stats(net_movies: list[nx.Graph],
                            movie_titles: list[str],
                            weight: str = "line_count",
                            comm_det_method: str = "LD") -> pd.DataFrame:
    columns = ["nodes", "edges", "max_degree", "density", "diameter", "assortativity", "avg_clustering",
               "avg_shortest_path",
               "transitivity", "number_connected_components", "number_of_cliques", "clique_number", "avg_rating",
               "num_votes",
               f"number_of_communities_{comm_det_method}", "gini_coef"]
    ratings = pd.read_csv("../data/imdb/movie_ratings.csv")
    measures = np.array([[nx.number_of_nodes(net) for net in net_movies],
                         [nx.number_of_edges(net) for net in net_movies],
                         [max_degree(net, weight) for net in net_movies],
                         [nx.density(net) for net in net_movies],
                         [nx.diameter(net) if nx.is_connected(net) else np.nan for net in net_movies],
                         [nx.degree_assortativity_coefficient(net, weight=weight) for net in net_movies],
                         [nx.average_clustering(net) for net in net_movies],
                         [nx.average_shortest_path_length(net, weight=weight) if nx.is_connected(net) else np.nan for
                          net in net_movies],
                         [nx.transitivity(net) for net in net_movies],
                         [nx.number_connected_components(net) for net in net_movies],
                         [nx.graph_number_of_cliques(net) for net in net_movies],
                         [nx.graph_clique_number(net) for net in net_movies],
                         [ratings.loc[ratings["title"] == title, "rating"].values[0] if ratings.loc[
                             ratings["title"] == title, "rating"].values else np.nan for title in movie_titles],
                         [ratings.loc[ratings["title"] == title, "num_votes"].values[0] if ratings.loc[
                             ratings["title"] == title, "num_votes"].values else np.nan for title in movie_titles],
                         [len(np.unique(detect_communities(net, method=comm_det_method, weight=weight))) for net in
                          net_movies],
                         [gini_coefficient(np.array(list(dict(net.degree(weight=weight)).values()))) for net in
                          net_movies]
                         ]).transpose()
    stats = pd.DataFrame(measures, index=movie_titles, columns=columns)
    return stats


def get_episode_dict(data_path: str) -> dict:
    data = pd.read_csv(data_path)
    seasons = data.season.unique()
    i = 0
    episode_dict = {}
    for season in seasons:
        office_raw_season = data[data.season == season]
        episodes = office_raw_season.episode.unique()
        for episode in episodes:
            episode_dict["s{0:02d}e{1:02d}".format(season, episode)] = i
            i += 1
    return episode_dict


def detect_communities(G: nx.Graph, method: str = "GM", weight: str = "line_count", resolution: float = 1.0,
                       seed: int = 777) -> list[
    int]:
    assert method.upper() in ["GM", "LV", "SG", "FG", "IM", "LE", "LP", "ML", "WT", "LD"]
    nodes = list(G.nodes())
    membership = list(np.zeros(len(nodes), dtype="int"))
    random.seed(seed)
    np.random.seed(seed)
    if method == "GM":
        communities = community.greedy_modularity_communities(G, weight=weight, resolution=resolution)
        com_dict = {character: i for i, com in enumerate(communities) for character in com}
        membership = [com_dict[c] for c in nodes]
    elif method == "LV":
        communities = community.louvain_communities(G, weight=weight, resolution=resolution)
        com_dict = {character: i for i, com in enumerate(communities) for character in com}
        membership = [com_dict[c] for c in nodes]
    elif method in ["SG", "FG", "IM", "LE", "LP", "ML", "WT", "LD"]:
        G_ig = ig.Graph.from_networkx(G)
        if method == "SG" and nx.is_connected(G):
            membership = G_ig.community_spinglass(weights=weight).membership
        elif method == "FG":
            membership = G_ig.community_fastgreedy(weights=weight).as_clustering().membership
        elif method == "IM":
            membership = G_ig.community_infomap(edge_weights=weight).membership
        elif method == "LE":
            try:
                membership = G_ig.community_leading_eigenvector(weights=weight).membership
            except ig._igraph.InternalError:
                print("Leading eigenvector error")
        elif method == "LP":
            membership = G_ig.community_label_propagation(weights=weight).membership
        elif method == "ML":
            membership = G_ig.community_multilevel(weights=weight).membership
        elif method == "WT":
            membership = G_ig.community_walktrap(weights=weight).as_clustering().membership
        elif method == "LD":
            membership = G_ig.community_leiden(weights=weight, n_iterations=-1, objective_function="modularity",
                                               resolution_parameter=resolution).membership
        else:
            return membership
    return membership


def draw_interaction_network_communities(G, weight=None, filename=None, resolution=1.0, method="LD",
                                         seed: int = None) -> None:
    '''
    Function that draws an interaction network from given graph
    :param G: networkx graph
    :param weight: name of edge attribute describing edge weight
    :param filename: name of the PNG file to which save the graph
    :param resolution: community detection parameter
    :param method: method of community detection (default greedy_modularity_communities)
    :param seed: Seed for the position of network drawing
    :return:
    '''
    nodes = list(G.nodes())
    edges = G.edges()
    if method:
        method = method.upper()
        colors = detect_communities(G, method=method, weight=weight, resolution=resolution)
    else:
        print("No community detection selected")
        colors = np.zeros(len(nodes))
    if weight is not None:
        degrees_weight = np.array([v for _, v in G.degree(weight=weight)])
        edge_width = np.array([G[u][v][weight] for u, v in edges])
        edge_width = edge_width / np.max(edge_width) * 8
    else:
        degrees_weight = np.array([v for _, v in G.degree()])
        edge_width = np.ones(len(edges))
    degrees_weight = degrees_weight / np.max(degrees_weight) * 4500
    # for i, component in enumerate(nx.connected_components(G)):
    #     sub_g = nx.subgraph(G, component)
    #     if method:
    #         method = method.upper()
    #         colors = detect_communities(sub_g, method=method, weight=weight, resolution=resolution)
    #     else:
    #         print("No community detection selected")
    #         colors = np.zeros(len(nodes))
    #
    pos = nx.spring_layout(G, seed=seed)
    fig, ax = plt.subplots(figsize=(12, 16))
    nx.draw_networkx_nodes(G, pos, node_size=degrees_weight, node_color=colors, cmap=plt.get_cmap("Set1"), alpha=0.9,
                           ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    plt.axis('off')
    # nx.draw_spring(G, with_labels=True, nodelist=nodes, node_size=degrees_weight, width=edge_width, node_color=colors, cmap=plt.get_cmap("Set1"))
    if filename:
        plt.savefig(f"../figures/{filename}.png")
        plt.close(fig)
    else:
        plt.show()


def plot_corr_mat(df: pd.DataFrame, filename: str = "", **kwargs) -> None:
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    df_corr = df.corr(**kwargs)
    plt.figure(figsize=(16, 9))
    sns.heatmap(df_corr,
                xticklabels=df_corr.columns.values,
                yticklabels=df_corr.columns.values,
                cmap=cmap,
                annot=True,
                fmt=".2f")
    plt.tight_layout()
    plt.title("Correlation of statistics")
    if filename:
        plt.savefig(f"../figures/{filename}.png")
        plt.close()
    else:
        plt.show()


def gini_coefficient(x: np.array) -> float:
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x) ** 2 * np.mean(x))


def create_similarity_matrix(episode_stats: pd.DataFrame, labels: list,
                             filename: str = "comparison/similarity_matrix") -> plt.Axes:
    norm_episode_stats = episode_stats.drop(["avg_rating", "num_votes", "runtime", "viewership", "show"], axis=1,
                                            errors="ignore").apply(
        lambda x: (x - x.mean()) / x.std(), axis=0)

    dists = pdist(norm_episode_stats.fillna(0), "euclidean")
    df_euclid = pd.DataFrame(1 / (1 + squareform(dists)), columns=norm_episode_stats.index,
                             index=norm_episode_stats.index)
    df_euclid.columns = labels
    df_euclid.index = labels
    cmap = sns.color_palette("light:b", as_cmap=True)
    fig = plt.figure(dpi=1200)
    ax = sns.heatmap(df_euclid,
                     xticklabels="auto",
                     yticklabels="auto",
                     cmap=cmap,
                     annot=False)
    plt.tight_layout()
    plt.savefig(f"../figures/{filename}")
    plt.close(fig)
    return ax


def get_community_detection_scores(show_name: str, weight: str = "line_count", methods: list = None, seed: int = 777) -> \
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if methods is None:
        methods = ["GM", "LV", "SG", "FG", "IM", "LE", "LP", "ML", "WT", "LD"]
    net_episodes = get_episode_networks(f"../data/{show_name}/")
    latest_file = [f for f in os.listdir(f"../data/{show_name}/") if f.startswith(f"{show_name}_lines_v")][-1]
    episode_dict = get_episode_dict(f"../data/{show_name}/{latest_file}")
    mod_df = pd.DataFrame(index=list(episode_dict.keys()))
    mix_pars_df = pd.DataFrame(index=list(episode_dict.keys()))
    num_communities_df = pd.DataFrame(index=list(episode_dict.keys()))
    for method in methods:
        print(method)
        mod_scores = []
        mix_pars = []
        num_communities = []
        for ep_code, ep_num in episode_dict.items():
            random.seed(seed)
            np.random.seed(seed)
            net_episode = net_episodes[ep_num]
            comm = detect_communities(net_episode, method=method, weight=weight)
            nodes = {i: x for i, x in enumerate(list(net_episode.nodes))}
            communities = [set(nodes[i] for i in np.nonzero(np.array(comm) == c)[0]) for c in np.unique(comm)]
            mod_score = nx.community.modularity(net_episode, communities, weight=weight)
            mod_scores.append(mod_score)
            # mixing parameter
            com_dict = {char: com for char, com in zip(list(net_episode.nodes), comm)}
            nx.set_node_attributes(net_episode, com_dict, "community")
            mix_par = mean_mixing_parameter(net_episode, "community")
            mix_pars.append(mix_par)
            # number of communities
            num_communities.append(len(np.unique(comm)))
        method_scores = pd.Series(mod_scores)
        method_scores.name = method
        method_scores.index = list(episode_dict.keys())
        mod_df = pd.concat([mod_df, method_scores], axis=1)
        mix_params = pd.Series(mix_pars)
        mix_params.name = method
        mix_params.index = list(episode_dict.keys())
        mix_pars_df = pd.concat([mix_pars_df, mix_params], axis=1)
        num_communities = pd.Series(num_communities)
        num_communities.name = method
        num_communities.index = list(episode_dict.keys())
        num_communities_df = pd.concat([num_communities_df, num_communities], axis=1)
    return mod_df, mix_pars_df, num_communities_df


def mean_mixing_parameter(g: nx.Graph, attribute: str, weight: str = "line_count"):
    nodes = list(g.nodes())
    nodes_data = g.nodes(data=True)
    mius = []
    for n in nodes:
        n_comm = nodes_data[n][attribute]
        subgraph = nx.subgraph(g, [character for character, data in nodes_data if data[attribute] != n_comm] + [n])
        k_ext = subgraph.degree(weight=weight)[n]
        k_tot = g.degree(weight=weight)[n]
        mius.append(k_ext / k_tot)
    if np.sum(mius) > 0:
        return np.mean(mius)
    else:
        return np.nan


def comm_det_test(net_episodes: list[nx.Graph], episode_dict: dict, method: str, weight: str = "line_count") -> None:
    for ep_code, ep_num in episode_dict.items():
        comm = detect_communities(net_episodes[ep_num], method=method, weight=weight)
