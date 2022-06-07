import matplotlib.pyplot as plt
import numpy as np

from utils import *
import os
import matplotlib
pd.options.display.max_columns = 30

matplotlib.use('Agg')

path_dir = "../data/movies"
movie_paths = [os.path.join(path_dir, movie) for movie in os.listdir(path_dir)]
movie_titles = [movie.split(".")[0] for movie in os.listdir(path_dir)]

os.makedirs("../figures/movies", exist_ok=True)

# i = 0
# for movie_path, movie_title in zip(movie_paths[i:], movie_titles[i:]):
#     print(movie_title)
#     edges_weighted = pd.read_csv(movie_path)
#     net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
#                                   edge_attr=["line_count", "scene_count", "word_count"])
#     draw_interaction_network_communities(net, "line_count", method="LD", filename=f"movies/{movie_title}_lines")
#     # draw_interaction_network_communities(net, "scene_count", method="LD", filename=f"movies/{movie_title}_scenes")
#     # draw_interaction_network_communities(net, "word_count", method="LD", filename=f"movies/{movie_title}_words")

selected_movies = ["batman", "blade_runner", "braveheart", "citizen_kane", "dead_poets_society", "die_hard", "fargo",
                   "good_will_hunting", "hannibal", "independence_day", "jaws_2", "jurassic_park",
                   "monty_python_and_the_holy_grail", "pirates_of_the_caribbean", "saving_private_ryan", "scream",
                   "spider-man", "superman", "the_big_lebowski", "the_bourne_identity",
                   "the_godfather", "the_matrix", "titanic", "tomorrow_never_dies"]

movies_net = []
for movie_path, movie_title in zip(movie_paths, movie_titles):
    if movie_title in selected_movies:
        print(movie_title)
        edges_weighted = pd.read_csv(movie_path)
        net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                                      edge_attr=["line_count", "scene_count", "word_count"])
        movies_net.append(net)

movie_stats = get_movie_network_stats(movies_net, selected_movies)


create_similarity_matrix(movie_stats, selected_movies, filename="comparison/movies")


# load serials
episode_stats = pd.DataFrame()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    net_episodes = get_episode_networks_limit(f"../data/{show_name}/", 5)
    latest_file = [f for f in os.listdir(f"../data/{show_name}/") if f.startswith(f"{show_name}_lines_v")][-1]
    episode_dict = get_episode_dict(f"../data/{show_name}/{latest_file}")
    ep_stats = get_network_stats_by_episode(net_episodes, episode_dict, show_name)
    ep_stats["show"] = show_name
    episode_stats = pd.concat([episode_stats, ep_stats], axis=0)

episode_stats.index = episode_stats["show"].values + "_" + episode_stats.index.values
episode_stats = episode_stats.drop(["runtime", "viewership", "show"], axis=1)

all_stats = pd.concat([movie_stats, episode_stats], axis=0)

# standardize all stats
norm_all_stats = all_stats.drop(["avg_rating", "num_votes"], axis=1, errors="ignore").apply(
        lambda x: (x - x.mean()) / x.std(), axis=0)

dists = pdist(norm_all_stats.fillna(0), "euclidean")
df_euclid = pd.DataFrame(1 / (1 + squareform(dists)), columns=norm_all_stats.index,
                         index=norm_all_stats.index)

most_similar = pd.DataFrame()
for movie in selected_movies:
    most_sim = df_euclid.iloc[:24, :].transpose().loc[:, movie].sort_values(ascending=False)[1:11]
    most_similar = pd.concat([most_similar, most_sim], axis=1)

most_sim_count = most_similar.notna().sum(axis=1).sort_values(ascending=False).head(20)

most_sim_count.index = most_sim_count.index.str.replace("_", " ").str.capitalize()
most_sim_count = most_sim_count.to_frame(name="count")

# similarity - movie vs series
most_similar_melt = df_euclid.iloc[:24, 24:].reset_index().melt(id_vars=["index"], var_name="tv_series", value_name="similarity")
most_similar_melt["index"] = most_similar_melt["index"].str.replace("_", " ").str.capitalize()
most_similar_melt["tv_series"] = most_similar_melt["tv_series"].str.replace("_", " ").str.capitalize()
most_similar_melt.sort_values("similarity", ascending=False).round(3).head(10)


df_euclid.iloc[:24, :].transpose().mean()
movies_similarities = (df_euclid.iloc[:24, :24].transpose().sum()-1)/24

serials_similarities = (df_euclid.iloc[:24, 24:].transpose().sum()-1)/813

pd.concat([movies_similarities, serials_similarities], axis=1)

matplotlib.use('Tkagg')

show_name = "the_office"
net_episodes = get_episode_networks_limit(f"../data/{show_name}/", 5)
latest_file = [f for f in os.listdir(f"../data/{show_name}/") if f.startswith(f"{show_name}_lines_v")][-1]
episode_dict = get_episode_dict(f"../data/{show_name}/{latest_file}")

draw_interaction_network_communities(net_episodes[episode_dict["s03e14"]], "line_count", method="LD", seed=777)

# PCA
from sklearn.decomposition import PCA
from adjustText import adjust_text

pca = PCA(n_components=4)
all_stats_pca = pca.fit_transform(norm_all_stats.dropna(axis=1))

print(pca.explained_variance_ratio_)

show_idx = dict()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    show_idx[show_name] = norm_all_stats.reset_index()[norm_all_stats.index.str.startswith(show_name)].index.values

fig, ax = plt.subplots()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    ax.scatter(all_stats_pca[show_idx[show_name], 0], all_stats_pca[show_idx[show_name], 2], label=f"{show_name} episodes", alpha=0.5)
ax.scatter(all_stats_pca[:24, 0], all_stats_pca[:24, 2], label="Movies", alpha=0.5)
# ts = []
# for x, y, text in zip(all_stats_pca[:24, 0], all_stats_pca[:24, 1], selected_movies):
#     ts.append(plt.text(x, y, text))
# adjust_text(ts, force_points=0.1, arrowprops=dict(arrowstyle='->', color='red'))
ax.legend()
plt.title("Principal component analysis of statistics of movies and TV series")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.show()


