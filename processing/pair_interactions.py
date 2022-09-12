import pandas as pd
from networks.utils import *

path = "../data/the_office"

merged_ep = pd.read_csv(f"{path}/merged_episodes_line_count.csv", index_col=[0, 1], header=[0, 1])
merged_ep.loc[('Jim', 'Michael')].rolling(10, min_periods=5, center=True).mean().plot(y="scene_count", figsize=(16, 9))

merged_seas = pd.read_csv(f"{path}/merged_seasons_line_count.csv", index_col=[0, 1])
merged_seas.loc[('Andy', 'Jim')].plot(kind="bar", rot=0)

idx = pd.IndexSlice
season_1 = merged_ep.loc[:, idx[["8","9"], :]].sum(axis=1)
season_1 = season_1[season_1 > 50].reset_index(name="line_count")
s1net = nx.from_pandas_edgelist(season_1, source="speaker1", target="speaker2", edge_attr=["line_count"])
draw_interaction_network_communities(s1net, "line_count", method="LD", seed=777)