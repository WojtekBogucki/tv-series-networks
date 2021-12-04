import pandas as pd
import numpy as np
import re

# load raw data
office_raw = pd.read_csv("../data/the_office/the_office_lines_v6.csv")

# filtering
speaker_count = office_raw.speaker.groupby(office_raw.speaker).count()
top_speakers = speaker_count[speaker_count > 100].index.tolist()
office_top_speakers = office_raw[office_raw.speaker.isin(top_speakers)]

office_top_speakers.head()
office_top_speakers.to_csv("../data/the_office/the_office_top_speakers.csv", index=False, encoding="utf-8")

line_count_by_scene = office_top_speakers.scene.groupby(office_top_speakers.scene).count()
group_scenes = line_count_by_scene[line_count_by_scene>1].index.tolist()
office_group_scenes = office_top_speakers[office_top_speakers.scene.isin(group_scenes)]
office_group_scenes.head()
office_group_scenes.to_csv("../data/the_office/the_office_group_scenes.csv", index=False, encoding="utf-8")

interactions = pd.DataFrame(columns=["speaker1", "speaker2", "line_count"])
office_count_by_scene_speaker = office_group_scenes.groupby(["scene", "speaker"])["line"].count().reset_index(name="count")
scenes = office_count_by_scene_speaker.scene.unique()
for scene in scenes:
    speakers_count = office_count_by_scene_speaker.loc[office_count_by_scene_speaker.scene == scene, ["speaker", "count"]].sort_values("speaker").reset_index(drop=True)
    n = speakers_count.shape[0]
    for i in range(n-1):
        for j in range(i+1, n):
            sp1 = speakers_count.iloc[i]
            sp2 = speakers_count.iloc[j]
            interactions = interactions.append({"speaker1": sp1["speaker"],
                                                "speaker2": sp2["speaker"],
                                                "line_count": sp1["count"] + sp2["count"]}, ignore_index=True)

interactions.head(20)

office_edges_weighted = interactions.groupby(["speaker1", "speaker2"])["line_count"].agg(line_count="sum",
                                                                                         scene_count="count").reset_index()
office_edges_weighted.head()
office_edges_weighted.to_csv("../data/the_office/the_office_edges_weighted.csv", index=False, encoding="utf-8")