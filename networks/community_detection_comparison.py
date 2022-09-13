from networks.utils import *
import pandas as pd
import os
from timeit import repeat
import numpy as np

# calculate modularity for each community detection method
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    print(show_name)
    mod_df, mix_par_df, num_com_def = get_community_detection_scores(show_name, seed=777)
    mod_df.to_csv(f"../data/communities/{show_name}_modularity.csv")
    mix_par_df.to_csv(f"../data/communities/{show_name}_mix_par.csv")
    num_com_def.to_csv(f"../data/communities/{show_name}_num_com.csv")

# average modularity
mods_df = pd.DataFrame()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    mod_df = pd.read_csv(f"../data/communities/{show_name}_modularity.csv", index_col=0)
    mods_df = pd.concat([mods_df, mod_df], axis=0, ignore_index=True)

mods_df.mean(axis=0).round(3)

# normalized
mods_df_norm = mods_df.div(mods_df.max(axis=1), axis=0).fillna(1)
mods_df_norm["SG"][mods_df_norm["SG"] < 0] = 0
mods_df_norm.mean(axis=0).sort_values(ascending=False).round(3)

# average modularity by show
mods_df = pd.DataFrame()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    mod_df = pd.read_csv(f"../data/communities/{show_name}_modularity.csv", index_col=0)
    mod_df_norm = mod_df.div(mod_df.max(axis=1), axis=0).fillna(1)
    mod_df_norm["SG"][mod_df_norm["SG"] < 0] = 0
    avg_mod = mod_df_norm.mean(axis=0)
    avg_mod = avg_mod.rename(show_name)
    mods_df = pd.concat([mods_df, avg_mod], axis=1)

mods_df.mean(axis=1).sort_values(ascending=False).round(3)

# average mixing parameter
mix_pars_df = pd.DataFrame()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    mix_par_df = pd.read_csv(f"../data/communities/{show_name}_mix_par.csv", index_col=0)
    mix_pars_df = pd.concat([mix_pars_df, mix_par_df], axis=0, ignore_index=True)

mix_pars_df.mean(axis=0).round(3)

# average number of communities
nums_com_def = pd.DataFrame()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    num_com_def = pd.read_csv(f"../data/communities/{show_name}_num_com.csv", index_col=0)
    nums_com_def = pd.concat([nums_com_def, num_com_def], axis=0, ignore_index=True)

num_com_def.mean(axis=0).round(3)

# example comparison - differences
mods_df = pd.DataFrame()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    mod_df = pd.read_csv(f"../data/communities/{show_name}_modularity.csv", index_col=0)
    mod_df = mod_df.reset_index()
    mod_df["show"]=show_name
    mods_df = pd.concat([mods_df, mod_df], axis=0, ignore_index=True)

# ML and LD
most_different = np.abs(mods_df["ML"] - mods_df["LD"]).sort_values(ascending=False).index.values[:10]
pd.options.display.max_columns=20
print(mods_df.loc[most_different, ["ML", "LD", "FG", "index", "show"]].round(3))

show_name = "the_office"
net_episodes = get_episode_networks(f"../data/{show_name}/")
latest_file = [f for f in os.listdir(f"../data/{show_name}/") if f.startswith(f"{show_name}_lines_v")][-1]
episode_dict = get_episode_dict(f"../data/{show_name}/{latest_file}")

draw_interaction_network_communities(net_episodes[episode_dict["s03e14"]], "line_count", method="LD", seed=777)
draw_interaction_network_communities(net_episodes[episode_dict["s03e14"]], "line_count", method="ML", seed=777)

mix_pars_df = pd.DataFrame()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    mix_par_df = pd.read_csv(f"../data/communities/{show_name}_mix_par.csv", index_col=0).reset_index()
    mix_par_df["show"] = show_name
    mix_pars_df = pd.concat([mix_pars_df, mix_par_df], axis=0, ignore_index=True)
most_different_mix_par = np.abs(mix_pars_df.dropna()["ML"] - mix_pars_df.dropna()["LD"]).sort_values(ascending=False).index.values[:10]
print(mix_pars_df.loc[most_different, ["ML", "LD", "FG", "index", "show"]])

# timing
setup = """
import random
import os
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from networkx import community
from networks.utils import get_episode_networks, get_episode_dict

show_name = "the_office"
net_episodes = get_episode_networks(f"../data/{show_name}/")
latest_file = [f for f in os.listdir(f"../data/{show_name}/") if f.startswith(f"{show_name}_lines_v")][-1]
episode_dict = get_episode_dict(f"../data/{show_name}/{latest_file}")     

ig_net_episodes = [ig.Graph.from_networkx(G) for G in net_episodes]                            

random.seed(123)
np.random.seed(123)
"""


nx_times = []
for method in ["greedy_modularity_communities", "louvain_communities"]:
    nx_stmt = f"""
    for ep_code, ep_num in episode_dict.items():
        communities = community.{method}(net_episodes[ep_num], weight="line_count")
    """
    nx_times.append(repeat(nx_stmt, setup=setup, number=10, repeat=5))
[np.mean(nx_time) for nx_time in nx_times]

ig_methods = ["community_spinglass(weights='line_count')",
               "community_fastgreedy(weights='line_count')",
               "community_infomap(edge_weights='line_count')",
               "community_leading_eigenvector(weights='line_count')",
               "community_label_propagation(weights='line_count')",
               "community_multilevel(weights='line_count')",
               "community_walktrap(weights='line_count')",
               "community_leiden(weights='line_count', n_iterations=-1, objective_function='modularity')"]

ig_times = []
for method in ig_methods:
    if "spinglass" in method:
        ig_stmt = f"""
for ep_code, ep_num in episode_dict.items():
    if nx.is_connected(net_episodes[ep_num]):
        communities = ig_net_episodes[ep_num].{method}
            """
    else:
        ig_stmt = f"""
for ep_code, ep_num in episode_dict.items():
    communities = ig_net_episodes[ep_num].{method}
        """
    ig_times.append(repeat(ig_stmt, setup=setup, number=10, repeat=5))

[np.mean(ig_time) for ig_time in ig_times]

times_df = pd.DataFrame(data=[np.mean(nx_time) for nx_time in nx_times]+[np.mean(ig_time) for ig_time in ig_times], index=["GM", "LV", "SG", "FG", "IM", "LE", "LP", "ML", "WT", "LD"], columns=["avg_time"])
times_df.to_csv("../data/communities/times.csv")


times_df = pd.read_csv("../data/communities/times.csv", index_col=0)
print((times_df/10).round(3))