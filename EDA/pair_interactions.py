import pandas as pd

path = "../data/the_office"

merged_ep = pd.read_csv(f"{path}/merged_episodes_line_count.csv", index_col=[0, 1], header=[0, 1])
merged_ep.loc[('Jim', 'Michael')].rolling(10, min_periods=5, center=True).mean().plot(y="scene_count", figsize=(16, 9))


merged_seas = pd.read_csv(f"{path}/merged_seasons_line_count.csv", index_col=[0, 1])
merged_seas.loc[('Andy', 'Jim')].plot(kind="bar", rot=0)
