import pandas as pd
import numpy as np
import re
import os

# load raw data
office_raw = pd.read_csv("../data/the_office/the_office_lines_v6.csv")


# filtering
def filter_by_speakers(dataset, count=100):
    speaker_count = dataset.speaker.groupby(dataset.speaker).count()
    top_speakers = speaker_count[speaker_count > count].index.tolist()
    return dataset[dataset.speaker.isin(top_speakers)]


def filter_group_scenes(dataset):
    line_count_by_scene = dataset.scene.groupby(dataset.scene).count()
    group_scenes = line_count_by_scene[line_count_by_scene > 1].index.tolist()
    return dataset[dataset.scene.isin(group_scenes)]


# transformation and aggregation
def get_speaker_network_edges(dataset):
    interactions = pd.DataFrame(columns=["speaker1", "speaker2", "line_count"])
    office_count_by_scene_speaker = dataset.groupby(["scene", "speaker"])["line"].count().reset_index(name="count")
    scenes = office_count_by_scene_speaker.scene.unique()
    for scene in scenes:
        speakers_count = office_count_by_scene_speaker.loc[office_count_by_scene_speaker.scene == scene,
                                                           ["speaker", "count"]].sort_values("speaker").reset_index(drop=True)
        n = speakers_count.shape[0]
        for i in range(n-1):
            for j in range(i+1, n):
                sp1 = speakers_count.iloc[i]
                sp2 = speakers_count.iloc[j]
                interactions = interactions.append({"speaker1": sp1["speaker"],
                                                    "speaker2": sp2["speaker"],
                                                    "line_count": sp1["count"] + sp2["count"]}, ignore_index=True)
    return interactions.groupby(["speaker1", "speaker2"])["line_count"].agg(line_count="sum",
                                                                            scene_count="count").reset_index()


# save speakers with over 100 lines
office_top_speakers = filter_by_speakers(office_raw)
office_top_speakers.to_csv("../data/the_office/the_office_top_speakers.csv", index=False, encoding="utf-8")

office_group_scenes = filter_group_scenes(office_top_speakers)
office_group_scenes.head()
office_group_scenes.to_csv("../data/the_office/the_office_group_scenes.csv", index=False, encoding="utf-8")

office_edges_weighted = get_speaker_network_edges(office_group_scenes)
office_edges_weighted.head()
office_edges_weighted.to_csv("../data/the_office/the_office_edges_weighted.csv", index=False, encoding="utf-8")

seasons = office_raw.season.unique()
for season in seasons:
    office_raw_season = office_raw[office_raw.season == season]
    season_edges = (office_raw_season.pipe(filter_by_speakers, count=20)
                                     .pipe(filter_group_scenes)
                                     .pipe(get_speaker_network_edges))
    season_edges.to_csv("../data/the_office/the_office_edges_weighted_S{0}.csv".format(season), index=False, encoding="utf-8")
    print("Season {} saved".format(season))

for season in seasons:
    office_raw_season = office_raw[office_raw.season == season]
    episodes = office_raw_season.episode.unique()
    dir_path = "../data/the_office/season{}".format(season)
    for episode in episodes:
        office_raw_episode = office_raw_season[office_raw_season.episode == episode]
        episode_edges = (office_raw_episode.pipe(filter_by_speakers, count=1)
                         .pipe(filter_group_scenes)
                         .pipe(get_speaker_network_edges))
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        episode_edges.to_csv(dir_path + "/the_office_edges_weighted_E{0}.csv".format(episode), index=False,
                             encoding="utf-8")
        print("Season {0} episode {1} saved".format(season, episode))