from EDA.processing import *

path = "../data/seinfeld"
seinfeld_raw = pd.read_csv(f"{path}/seinfeld_lines_v2.csv", encoding="utf-8")

# save speakers with over 100 lines
edges_weighted = (seinfeld_raw.pipe(filter_by_speakers)
                  .pipe(filter_group_scenes)
                  .pipe(get_speaker_network_edges))
edges_weighted.to_csv(f"{path}/edges_weighted.csv", index=False, encoding="utf-8")

# top 30 speakers
edges_weighted = (seinfeld_raw.pipe(filter_by_speakers, top=30)
                  .pipe(filter_group_scenes)
                  .pipe(get_speaker_network_edges))
edges_weighted.to_csv(f"{path}/edges_weighted_top30.csv", index=False, encoding="utf-8")

save_seasons(seinfeld_raw, path=path, count=20)
save_episodes(seinfeld_raw, path=path, count=0)
