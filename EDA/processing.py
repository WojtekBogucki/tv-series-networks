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


def fix_filtered_names(dataset, episodes, replacements):
    filter_ep = (dataset.season == episodes[0][0]) & (dataset.episode == episodes[0][1])
    if len(episodes) > 1:
        for episode in episodes[1:]:
            filter_ep = filter_ep | (dataset.season == episode[0]) & (dataset.episode == episode[1])
    dataset.loc[filter_ep, 'speaker'] = dataset.loc[filter_ep, 'speaker'].apply(fix_names, args=(replacements,))
    return dataset


def distinguish_characters(dataset: pd.DataFrame, characters: list):
    '''
    Distinguish generic characters e.g. "Man", "Woman" by adding season and episode
    :param dataset:
    :param characters:
    :return:
    '''
    for character in characters:
        dataset.loc[dataset.speaker == character, "speaker"] = dataset[dataset.speaker == character].apply(
            lambda x: f"{character} s{x.season:02d}e{x.episode:02d}", axis=1)
    return dataset


def split_characters(dataset, splitters):
    '''
    Split multiple characters e.g. "Michael and Dwight" to separate records with duplicated lines
    :param dataset: Pandas Data Frame
    :param splitters: list of strings
    :return: Data Frame with splitted characters
    '''
    for splitter in splitters:
        filter_speakers = dataset.speaker.str.strip().str.contains(splitter)
        dataset.loc[filter_speakers, "speaker"] = dataset.speaker[filter_speakers].apply(
            lambda x: x.split(splitter))
        dataset = dataset.explode("speaker")
    return dataset


def remove_speakers(dataset: pd.DataFrame, speakers: list):
    '''
    Remove speakers from dataset
    :param dataset: pandas Data Frame
    :param speakers: list of speakers to remove
    :return: pandas Data Frame
    '''
    dataset = dataset.reset_index(drop=True)
    dataset = dataset.drop(dataset[dataset.speaker.isin(speakers)].index).reset_index(drop=True)
    return dataset


def filter_by_speakers(dataset, count=100, top=None):
    speaker_count = dataset.speaker.groupby(dataset.speaker).count()
    if top is None:
        top_speakers = speaker_count[speaker_count > count].index.tolist()
    else:
        top_speakers = speaker_count.sort_values(ascending=False)[:top].index.to_list()
    return dataset[dataset.speaker.isin(top_speakers)]


def filter_group_scenes(dataset):
    line_count_by_scene = dataset.scene.groupby(dataset.scene).count()
    group_scenes = line_count_by_scene[line_count_by_scene > 1].index.tolist()
    return dataset[dataset.scene.isin(group_scenes)]


# transformation and aggregation
def get_speaker_network_edges(dataset):
    interactions = pd.DataFrame(columns=["speaker1", "speaker2", "line_count", "word_count"])
    word_count = dataset.apply(lambda x: len(re.split(r" |'", str(x.line))), axis=1)
    dataset = dataset.assign(word_count=word_count)
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
