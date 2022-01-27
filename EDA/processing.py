import re
import pandas as pd
import os

def fix_names(x, replacements):
    '''
    Fix characters' names
    :param x: pandas Series
    :param replacements: dict
    :return:
    '''
    x = x.strip().replace(":", "")
    for rep in replacements:
        x = re.sub(r'^'+rep+'$', replacements[rep], x)
    return x


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
    interactions = pd.DataFrame(columns=["speaker1", "speaker2", "line_count", "word_count"])
    dataset["word_count"] = dataset.loc[:, "line"].apply(lambda x: len(str(x).split(" ")))
    office_count_by_scene_speaker = dataset.groupby(["scene", "speaker"]).agg(word_count=("word_count", "sum"),
                                                                              line=("line", "count")).reset_index()
    # word_count_by_scene_speaker = dataset.groupby(["scene", "speaker"])["line"].count().reset_index(name="word_count")
    scenes = office_count_by_scene_speaker.scene.unique()
    for scene in scenes:
        speakers_count = office_count_by_scene_speaker.loc[office_count_by_scene_speaker.scene == scene,
                                                           ["speaker", "word_count", "line"]].sort_values("speaker").reset_index(drop=True)
        n = speakers_count.shape[0]
        for i in range(n-1):
            for j in range(i+1, n):
                sp1 = speakers_count.iloc[i]
                sp2 = speakers_count.iloc[j]
                interactions = interactions.append({"speaker1": sp1["speaker"],
                                                    "speaker2": sp2["speaker"],
                                                    "line_count": sp1["line"] + sp2["line"],
                                                    "word_count": sp1["word_count"] + sp2["word_count"]}, ignore_index=True)
    return interactions.groupby(["speaker1", "speaker2"]).agg(line_count=("line_count", "sum"),
                                                              scene_count=("line_count", "count"),
                                                              word_count=("word_count", "sum")).reset_index()


def save_seasons(dataset, count=20, path="../data"):
    seasons = dataset.season.unique()
    for season in seasons:
        raw_season = dataset[dataset.season == season]
        season_edges = (raw_season.pipe(filter_by_speakers, count=count)
                                  .pipe(filter_group_scenes)
                                  .pipe(get_speaker_network_edges))
        season_edges.to_csv(f"{path}/edges_weighted_S{season}.csv", index=False, encoding="utf-8")
        print(f"Season {season} saved")


def save_episodes(dataset, count=1, path="../data"):
    seasons = dataset.season.unique()
    for season in seasons:
        raw_season = dataset[dataset.season == season]
        episodes = raw_season.episode.unique()
        dir_path = f"{path}/season{season}"
        for episode in episodes:
            raw_episode = raw_season[raw_season.episode == episode]
            episode_edges = (raw_episode.pipe(filter_by_speakers, count=count)
                             .pipe(filter_group_scenes)
                             .pipe(get_speaker_network_edges))
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            episode_edges.to_csv(dir_path + f"/edges_weighted_E{episode:02d}.csv", index=False,
                                 encoding="utf-8")
            print(f"Season {season} episode {episode} saved")
