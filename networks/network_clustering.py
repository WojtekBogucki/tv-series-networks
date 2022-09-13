from networks.utils import *
import os
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics.cluster import contingency_matrix

import numpy as np

pd.options.display.max_columns = 30


episode_stats = pd.DataFrame()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    net_episodes = get_episode_networks(f"../data/{show_name}/")
    latest_file = [f for f in os.listdir(f"../data/{show_name}/") if f.startswith(f"{show_name}_lines_v")][-1]
    episode_dict = get_episode_dict(f"../data/{show_name}/{latest_file}")
    ep_stats = get_network_stats_by_episode(net_episodes, episode_dict, show_name)
    ep_stats["show"] = show_name
    episode_stats = pd.concat([episode_stats, ep_stats], axis=0)

episode_stats.index = episode_stats["show"].values + "_" + episode_stats.index.values

norm_episode_stats = episode_stats.drop(["avg_rating", "num_votes", "runtime", "viewership", "show"], axis=1,
                                            errors="ignore").apply(lambda x: (x - x.mean()) / x.std(), axis=0)

clustering = AgglomerativeClustering(compute_full_tree=True, n_clusters=5).fit(norm_episode_stats.fillna(0))
clustering

clustering.labels_

episode_stats.index.values[clustering.labels_==2]

episode_stats["cluster"] = clustering.labels_

episode_stats.groupby("cluster").mean()

episode_stats.groupby(["cluster", "show"]).size()
cm = contingency_matrix(episode_stats["show"].values, clustering.labels_)
pd.DataFrame(cm, index=["friends", "seinfeld", "tbbt", "the office"], columns=np.unique(clustering.labels_))


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


model = AgglomerativeClustering(compute_full_tree=True, n_clusters=None, distance_threshold=0).fit(norm_episode_stats.fillna(0))
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()