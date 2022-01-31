import pandas as pd
import numpy as np
import re
import os
from EDA.processing import filter_by_speakers, filter_group_scenes, get_speaker_network_edges, save_seasons, save_episodes

# load raw data
office_raw = pd.read_csv("../data/the_office/the_office_lines_v6.csv")

# save speakers with over 100 lines
office_top_speakers = filter_by_speakers(office_raw)
office_top_speakers.to_csv("../data/the_office/the_office_top_speakers.csv", index=False, encoding="utf-8")

office_group_scenes = filter_group_scenes(office_top_speakers)
office_group_scenes.head()
office_group_scenes.to_csv("../data/the_office/the_office_group_scenes.csv", index=False, encoding="utf-8")

office_edges_weighted = get_speaker_network_edges(office_group_scenes)
office_edges_weighted.head()
office_edges_weighted.to_csv("../data/the_office/the_office_edges_weighted.csv", index=False, encoding="utf-8")

# top 30 speakers
office_top_speakers = filter_by_speakers(office_raw, top=30)
office_top_speakers.to_csv("../data/the_office/top_speakers_top30.csv", index=False, encoding="utf-8")

office_group_scenes = filter_group_scenes(office_top_speakers)
office_group_scenes.head()
office_group_scenes.to_csv("../data/the_office/group_scenes_top30.csv", index=False, encoding="utf-8")

office_edges_weighted = get_speaker_network_edges(office_group_scenes)
office_edges_weighted.head()
office_edges_weighted.to_csv("../data/the_office/edges_weighted_top30.csv", index=False, encoding="utf-8")


save_seasons(office_raw, path="../data/the_office")
save_episodes(office_raw, path="../data/the_office")

test_group = office_raw[(office_raw.season==3) & (office_raw.episode==20)].groupby('scene')
scenes_group = {scene: list(test_group.get_group(name=scene).speaker) for scene in test_group.groups}
import json
with open('../data/s03ep20_speakers.json', 'w+') as f:
    json.dump(scenes_group, f, indent=4)

test_line = office_raw.line[0]
office_raw["word_count"] = office_raw.loc[:, "line"].apply(lambda x: len(str(x).split(" ")))
office_raw.groupby(["scene", "speaker"])["word_count", "line"].agg({"word_count": "sum","line": "count"}).reset_index()

(office_raw.iloc[0:100,:].pipe(filter_by_speakers, count=1)
                                  .pipe(filter_group_scenes)
                                  .pipe(get_speaker_network_edges))