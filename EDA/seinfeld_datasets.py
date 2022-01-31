import pandas as pd
import numpy as np
import re
import os
from EDA.processing import filter_by_speakers, filter_group_scenes, get_speaker_network_edges, save_seasons, save_episodes

path = "../data/seinfeld"
seinfeld_raw = pd.read_csv(f"{path}/seinfeld_lines_v2.csv", encoding="utf-8")

# save speakers with over 100 lines
top_speakers = filter_by_speakers(seinfeld_raw)
top_speakers.to_csv(f"{path}/top_speakers.csv", index=False, encoding="utf-8")

group_scenes = filter_group_scenes(top_speakers)
group_scenes.head()
group_scenes.to_csv(f"{path}/group_scenes.csv", index=False, encoding="utf-8")

edges_weighted = get_speaker_network_edges(group_scenes)
edges_weighted.head()
edges_weighted.to_csv(f"{path}/edges_weighted.csv", index=False, encoding="utf-8")

# top 30 speakers
top_speakers = filter_by_speakers(seinfeld_raw, top=30)
top_speakers.to_csv(f"{path}/top_speakers_top30.csv", index=False, encoding="utf-8")

group_scenes = filter_group_scenes(top_speakers)
group_scenes.head()
group_scenes.to_csv(f"{path}/group_scenes_top30.csv", index=False, encoding="utf-8")

edges_weighted = get_speaker_network_edges(group_scenes)
edges_weighted.head()
edges_weighted.to_csv(f"{path}/edges_weighted_top30.csv", index=False, encoding="utf-8")

save_seasons(seinfeld_raw, path=path)
save_episodes(seinfeld_raw, path=path)