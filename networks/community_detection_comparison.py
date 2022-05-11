from networks.utils import *
import pandas as pd
import os
from timeit import timeit
import numpy
import random
import json

# calculate time and modularity for each community detection method
times_df = pd.DataFrame()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    print(show_name)
    mod_df, times = get_community_detection_scores(show_name)
    mod_df.to_csv(f"../data/communities/{show_name}_modularity.csv")
    times_df = pd.concat([times_df, pd.Series(times, name=show_name)], axis=1)

times_df.to_csv("../data/communities/times.csv")

times_df = pd.read_csv("../data/communities/times.csv", index_col=0)
times_df.mean(axis=1).sort_values()

times_df_norm = times_df/times_df.max(axis=0)
times_df_norm.mean(axis=1).sort_values()

mods_df = pd.DataFrame()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    mod_df = pd.read_csv(f"../data/communities/{show_name}_modularity.csv", index_col=0)
    mod_df_norm = mod_df.div(mod_df.max(axis=1), axis=0).fillna(1)
    mod_df_norm["SG"][mod_df_norm["SG"] < 0] = 0
    avg_mod = mod_df_norm.mean(axis=0)
    avg_mod = avg_mod.rename(show_name)
    mods_df = pd.concat([mods_df, avg_mod], axis=1)

mods_df.mean(axis=1).sort_values(ascending=False)

# example comparison
show_name = "tbbt"
net_episodes = get_episode_networks(f"../data/{show_name}/")
latest_file = [f for f in os.listdir(f"../data/{show_name}/") if f.startswith(f"{show_name}_lines_v")][-1]
episode_dict = get_episode_dict(f"../data/{show_name}/{latest_file}")

draw_interaction_network_communities(net_episodes[episode_dict["s01e01"]], "line_count", method="LD")
draw_interaction_network_communities(net_episodes[episode_dict["s01e01"]], "line_count", method="LV")





# timeit
times = dict()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    random.seed(123)
    numpy.random.seed(123)
    net_episodes = get_episode_networks(f"../data/{show_name}/")
    latest_file = [f for f in os.listdir(f"../data/{show_name}/") if f.startswith(f"{show_name}_lines_v")][-1]
    episode_dict = get_episode_dict(f"../data/{show_name}/{latest_file}")
    serial_times = dict()
    for method in ["LD", "FG", "ML", "GM", "LV"]:
        serial_times[method] = timeit(lambda: comm_det_test(net_episodes, episode_dict, method), number=100)

    print("\n".join(f"{key}: {value:.3f}" for key, value in serial_times.items()))
    times[show_name] = serial_times


with open("../data/communities/timeit.json", "w") as f:
    json.dump(times, f, indent=4)
