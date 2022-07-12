import re
import pandas as pd
import os

from matplotlib import pyplot as plt


def fix_names(x, replacements):
    '''
    Fix characters' names
    :param x: pandas Series
    :param replacements: dict
    :return:
    '''
    x = x.strip().replace(":", "")
    for rep in replacements:
        x = re.sub(r'^' + rep + '$', replacements[rep], x)
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
    # interactions = pd.DataFrame(columns=["speaker1", "speaker2", "line_count", "word_count"])
    word_count = dataset.apply(lambda x: len(re.split(r" |'", str(x.line))), axis=1)
    dataset = dataset.assign(word_count=word_count)
    office_count_by_scene_speaker = dataset.groupby(["scene", "speaker"]).agg(word_count=("word_count", "sum"),
                                                                              line=("line", "count")).reset_index()
    # word_count_by_scene_speaker = dataset.groupby(["scene", "speaker"])["line"].count().reset_index(name="word_count")
    scenes = office_count_by_scene_speaker.scene.unique()
    speaker1 = []
    speaker2 = []
    line_count = []
    word_count = []
    for scene in scenes:
        speakers_count = office_count_by_scene_speaker.loc[office_count_by_scene_speaker.scene == scene,
                                                           ["speaker", "word_count", "line"]].sort_values(
            "speaker").reset_index(drop=True)
        n = speakers_count.shape[0]
        for i in range(n - 1):
            for j in range(i + 1, n):
                sp1 = speakers_count.iloc[i]
                sp2 = speakers_count.iloc[j]
                speaker1.append(sp1["speaker"])
                speaker2.append(sp2["speaker"])
                line_count.append(sp1["line"] + sp2["line"])
                word_count.append(sp1["word_count"] + sp2["word_count"])
    interactions = pd.concat([pd.Series(speaker1, name="speaker1"),
                              pd.Series(speaker2, name="speaker2"),
                              pd.Series(line_count, name="line_count"),
                              pd.Series(word_count, name="word_count")], axis=1)
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


def save_merged_episodes(path: str = "../data") -> None:
    seasons = [os.path.join(path, dirname) for dirname in os.listdir(path) if
               os.path.isdir(os.path.join(path, dirname)) and dirname.startswith("season")]
    df = pd.DataFrame(columns=["speaker1", "speaker2", "line_count", "word_count", "scene_count", "season", "episode"])
    pattern = re.compile(r"edges_weighted_E(\d+)\.csv")
    for i, season in enumerate(seasons):
        episodes = os.listdir(season)
        for episode in episodes:
            ep_df = pd.read_csv(os.path.join(season, episode))
            ep_df["season"] = i + 1
            ep_df["episode"] = int(pattern.search(episode).group(1))
            df = pd.concat([df, ep_df], axis=0)
    for measure in ["line_count", "word_count", "scene_count"]:
        pivot_df = pd.pivot_table(df, values=measure, index=["speaker1", "speaker2"], columns=["season", "episode"],
                                  fill_value=0)
        pivot_df.to_csv(os.path.join(path, f"merged_episodes_{measure}.csv"), encoding="utf-8")


def merge_seasons(path: str) -> None:
    num_seasons = len(
        [f for f in os.listdir(path) if f.startswith("edges_weighted_S") and os.path.isfile(os.path.join(path, f))])
    df = pd.DataFrame(columns=["speaker1", "speaker2", "line_count", "word_count", "scene_count", "season"])
    for i in range(num_seasons):
        season_df = pd.read_csv(os.path.join(path, f"edges_weighted_S{i + 1}.csv"))
        season_df["season"] = i + 1
        df = pd.concat([df, season_df], axis=0)
    for measure in ["line_count", "word_count", "scene_count"]:
        pivot_df = pd.pivot_table(df, values=measure, index=["speaker1", "speaker2"], columns="season", fill_value=0)
        pivot_df.to_csv(os.path.join(path, f"merged_seasons_{measure}.csv"), encoding="utf-8")


def get_valid_filename(s: str) -> str:
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w]', '', s)


def visualize_eda(data: pd.DataFrame, show_name: str) -> None:
    os.makedirs(f"../figures/{show_name}/eda", exist_ok=True)
    lines_by_season = data.groupby('season')['line'].count()
    episodes_by_season = data.groupby('season')['episode'].nunique()
    lines_per_episode_by_season = lines_by_season / episodes_by_season
    scenes_by_season = data.groupby('season')['scene'].nunique()
    scenes_per_episode_by_season = scenes_by_season / episodes_by_season
    scenes_by_season_episode = data.groupby(['season', 'episode'])['scene'].nunique().reset_index(name="count")
    lines_by_speaker = data.groupby(['speaker', 'season'])['line'].size().unstack(fill_value=0)
    lines_by_speaker["sum_col"] = lines_by_speaker.sum(axis=1)
    lines_by_speaker = lines_by_speaker.sort_values(by="sum_col", ascending=False)
    lines_by_speaker = lines_by_speaker.drop("sum_col", axis=1)
    episodes_by_speaker = data.loc[:, ['speaker', 'season', 'episode']].drop_duplicates().groupby(['speaker'])[
                              'speaker'].count().sort_values(ascending=False)[:15]
    episodes_by_number_of_speaker = \
        data.loc[:, ['speaker', 'season', 'episode']].drop_duplicates().groupby(['season', 'episode'])[
            'speaker'].count().sort_values(ascending=False)[:20]

    fig, ax = plt.subplots()
    lines_by_season.plot.bar(title="Number of lines by season", ylabel="Number of lines", ax=ax)
    plt.xticks(rotation=0)
    plt.savefig(f"../figures/{show_name}/eda/{show_name}_lines_by_season.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    lines_per_episode_by_season.plot.bar(title="Average number of lines per episode by season",
                                         ylabel="Average number of lines per episode", ax=ax)
    plt.xticks(rotation=0)
    plt.savefig(f"../figures/{show_name}/eda/{show_name}_lines_per_episode_by_season.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    scenes_by_season.plot.bar(title="Number of scenes by season", ylabel="Number of scenes", ax=ax)
    plt.xticks(rotation=0)
    plt.savefig(f"../figures/{show_name}/eda/{show_name}_scenes_by_season.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    scenes_per_episode_by_season.plot.bar(title="Average number of scenes per episode by season",
                                          ylabel="Average number of scenes per episode", ax=ax)
    plt.xticks(rotation=0)
    plt.savefig(f"../figures/{show_name}/eda/{show_name}_scenes_per_episode_by_season.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    scenes_by_season_episode.boxplot(column="count", by="season", ax=ax)
    plt.title("Number of scenes")
    plt.ylabel("Number of scenes per episode")
    plt.xticks(rotation=0)
    plt.savefig(f"../figures/{show_name}/eda/{show_name}_scenes_per_episode_boxplot.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    lines_by_speaker[:15].plot(kind="bar", stacked=True, colormap="inferno", title="Lines spoken by character",
                               ylabel="Number of lines", figsize=(12, 8), ax=ax)
    plt.xticks(rotation=45)
    plt.savefig(f"../figures/{show_name}/eda/{show_name}_speakers_by_season.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    episodes_by_speaker.plot(kind="bar", title="Number of episodes in which characters occurred",
                             ylabel="Number of episodes", figsize=(12, 8), ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"../figures/{show_name}/eda/{show_name}_episodes_by_speaker.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    episodes_by_number_of_speaker.plot(kind="bar", title="Number of characters in episodes",
                                       ylabel="Number of characters", figsize=(12, 8), ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"../figures/{show_name}/eda/{show_name}_episodes_by_number_of_speaker.png")
    plt.close(fig)
