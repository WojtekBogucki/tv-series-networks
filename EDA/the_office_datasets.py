import pandas as pd
from EDA.processing import *

path = "../data/the_office"

# load raw data
office_raw = pd.read_csv(f"{path}/the_office_lines_v6.csv")

# save speakers with over 100 lines
office_edges_weighted = (office_raw.pipe(filter_by_speakers)
                         .pipe(filter_group_scenes)
                         .pipe(get_speaker_network_edges))
office_edges_weighted.to_csv(f"{path}/edges_weighted.csv", index=False, encoding="utf-8")

# top 30 speakers
top30_office_edges_weighted = (office_raw.pipe(filter_by_speakers, top=30)
                               .pipe(filter_group_scenes)
                               .pipe(get_speaker_network_edges))
top30_office_edges_weighted.to_csv(f"{path}/edges_weighted_top30.csv", index=False, encoding="utf-8")

save_seasons(office_raw, count=20, path=path)
save_episodes(office_raw, count=0, path=path)

save_merged_episodes(path)
merged_ep = pd.read_csv(f"{path}/merged_episodes_line_count.csv", index_col=[0, 1], header=[0, 1])
merged_ep.loc[('Andy', 'Jim')].rolling(50, min_periods=1, center=True).mean().plot(y="line_count", figsize=(16,9))

merge_seasons(path)
merged_seas = pd.read_csv(f"{path}/merged_seasons_line_count.csv", index_col=[0, 1])
merged_seas.loc[('Andy', 'Jim')].plot(kind="bar", rot=0)

#############
# test_group = office_raw[(office_raw.season == 3) & (office_raw.episode == 20)].groupby('scene')
# scenes_group = {scene: list(test_group.get_group(name=scene).speaker) for scene in test_group.groups}
# import json
#
# with open('../data/s03ep20_speakers.json', 'w+') as f:
#     json.dump(scenes_group, f, indent=4)
#
# test_line = office_raw.line[0]
# office_raw["word_count"] = office_raw.loc[:, "line"].apply(lambda x: len(str(x).split(" ")))
# office_raw.groupby(["scene", "speaker"])["word_count", "line"].agg({"word_count": "sum", "line": "count"}).reset_index()
#
# (office_raw.iloc[0:100, :].pipe(filter_by_speakers, count=1)
#  .pipe(filter_group_scenes)
#  .pipe(get_speaker_network_edges))
