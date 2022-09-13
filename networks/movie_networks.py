from utils import *
import os
import matplotlib
from sklearn.decomposition import PCA
from adjustText import adjust_text
pd.options.display.max_columns = 30

matplotlib.use('Agg')

selected_movies = ["batman", "blade_runner", "braveheart", "citizen_kane", "dead_poets_society", "die_hard", "fargo",
                   "good_will_hunting", "hannibal", "independence_day", "jaws_2", "jurassic_park",
                   "monty_python_and_the_holy_grail", "pirates_of_the_caribbean", "saving_private_ryan", "scream",
                   "spider-man", "superman", "the_big_lebowski", "the_bourne_identity",
                   "the_godfather", "the_matrix", "titanic", "tomorrow_never_dies"]


def draw_movies_networks(selected: list, path_dir: str = "../data/movies") -> None:
    os.makedirs("../figures/movies", exist_ok=True)
    paths = [os.path.join(path_dir, movie) for movie in os.listdir(path_dir)]
    titles = [movie.split(".")[0] for movie in os.listdir(path_dir)]
    for movie_path, movie_title in zip(paths, titles):
        if movie_title in selected:
            print(movie_title)
            edges_weighted = pd.read_csv(movie_path)
            net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                                          edge_attr=["line_count", "scene_count", "word_count"])
            draw_interaction_network_communities(net, "line_count", method="LD", filename=f"movies/{movie_title}_lines")
            # draw_interaction_network_communities(net, "scene_count", method="LD", filename=f"movies/{movie_title}_scenes")
            # draw_interaction_network_communities(net, "word_count", method="LD", filename=f"movies/{movie_title}_words")


def get_movies_networks(selected: list, path_dir: str = "../data/movies") -> list:
    paths = [os.path.join(path_dir, movie) for movie in os.listdir(path_dir)]
    titles = [movie.split(".")[0] for movie in os.listdir(path_dir)]
    movies_net = []
    for movie_path, movie_title in zip(paths, titles):
        if movie_title in selected:
            print(movie_title)
            edges_weighted = pd.read_csv(movie_path)
            net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                                          edge_attr=["line_count", "scene_count", "word_count"])
            movies_net.append(net)
    return movies_net


movies_net = get_movies_networks(selected_movies)
movie_stats = get_movie_network_stats(movies_net, selected_movies)


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

# the most similar
most_similar = pd.DataFrame()
for movie in selected_movies:
    most_sim = df_euclid.iloc[:24, :].transpose().loc[:, movie].sort_values(ascending=False)[1:11]
    most_similar = pd.concat([most_similar, most_sim], axis=1)

most_sim_count = most_similar.notna().sum(axis=1).sort_values(ascending=False).head(20)

most_sim_count.index = most_sim_count.index.str.replace("_", " ").str.capitalize()
most_sim_count = most_sim_count.to_frame(name="count")
print(most_sim_count)

# similarity - movie vs series
most_similar_melt = df_euclid.iloc[:24, 24:].reset_index().melt(id_vars=["index"], var_name="tv_series", value_name="similarity")
most_similar_melt["index"] = most_similar_melt["index"].str.replace("_", " ").str.capitalize()
most_similar_melt["tv_series"] = most_similar_melt["tv_series"].str.replace("_", " ").str.capitalize()
most_similar_melt.sort_values("similarity", ascending=False).round(3).head(10)

# movies and serials average similarity
df_euclid.iloc[:24, :].transpose().mean()
movies_similarities = (df_euclid.iloc[:24, :24].transpose().sum()-1)/24

serials_similarities = (df_euclid.iloc[:24, 24:].transpose().sum()-1)/813

pd.concat([movies_similarities, serials_similarities], axis=1)

# the most unique
avg_similarity = (df_euclid.sum(axis=1)-1)/(df_euclid.shape[0]-1)

most_unique = avg_similarity.sort_values(ascending=True).round(3).head(10)
indexes = []
for show_name in most_unique.index.values:
    indexes.append(norm_all_stats.reset_index()[norm_all_stats.index.str.startswith(show_name)].index.values[0])

most_unique.index = most_unique.index.str.split("_").map(lambda x: " ".join(e.capitalize() for e in x))


matplotlib.use('Tkagg')

# PCA
pca = PCA(n_components=2)
print(norm_all_stats.isna().sum())
norm_all_stats_clean = norm_all_stats.dropna(axis=1)
all_stats_pca = pca.fit_transform(norm_all_stats_clean)
loadings = pd.DataFrame(pca.components_, columns=pca.feature_names_in_, index=["Component 1", "Component 2"])

fig, ax = plt.subplots(figsize=(8,5))
loadings.transpose().plot(kind="barh", layout=(1, 2), subplots=False, sharey=True, legend=True, ax=ax)
plt.xlabel("Loadings")
plt.tight_layout()
plt.savefig("../figures/comparison/loadings_all")

print(pca.explained_variance_ratio_)

show_idx = dict()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    show_idx[show_name] = norm_all_stats.reset_index()[norm_all_stats.index.str.startswith(show_name)].index.values

show_names = {
    "the_office": "The Office",
    "seinfeld": "Seinfeld",
    "tbbt": "The Big Bang Theory",
    "friends": "Friends"
}

alpha = 0.8
s = 50
plt.style.use('default')
_, ax = plt.subplots(figsize=(16, 8))
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    ax.scatter(all_stats_pca[show_idx[show_name], 0], all_stats_pca[show_idx[show_name], 1], label=f"{show_names[show_name]} episodes", alpha=alpha, s=s)
ax.scatter(all_stats_pca[:24, 0], all_stats_pca[:24, 1], label="Movies", alpha=alpha, s=s)
text = [plt.text(all_stats_pca[idx, 0], all_stats_pca[idx, 1], title, fontsize=12) for title, idx in zip(most_unique.index.values, indexes)]
ax.legend()
plt.title("Principal component analysis of statistics of movies and TV series", fontsize=16)
plt.xlabel(f"Principal component 1 ({100*pca.explained_variance_ratio_[0]:.2f} % explained variance)", fontsize=12)
plt.ylabel(f"Principal component 2 ({100*pca.explained_variance_ratio_[1]:.2f} % explained variance)", fontsize=12)
adjust_text(text, x=all_stats_pca[:, 0], y=all_stats_pca[:, 1], arrowprops=dict(arrowstyle='->', color='red'))
plt.savefig("../figures/comparison/pca_all")


