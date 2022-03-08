import pandas as pd
import matplotlib.pyplot as plt
from EDA.processing import fix_names, split_characters

pd.options.display.max_columns = 10
pd.options.display.max_rows = None

friends_df = pd.read_csv("../data/friends/friends_lines_v1.csv", encoding="utf-8")
len(friends_df.speaker[friends_df.speaker.str.contains(" and ")])
len(friends_df.speaker[friends_df.speaker.str.contains(", ")])
len(friends_df.speaker[friends_df.speaker.str.contains(" & ")])

friends_df = split_characters(friends_df, [" and ", ", ", " & "])
friends_df[friends_df.season==1].groupby("speaker").size().reset_index(name="count").sort_values("speaker")

replacements = {"mr\.heckles": "mr. heckles",
                "rach": "rachel",
                "mnca": "monica",
                "chan": "chandler",
                "phoe": "phoebe",
                "mich": "michael",
                "rtst": "mr. ratstatter",
                "fbob": "fun bobby",
                "estl": "estelle",
                "the director": "director",
                "mr zelner": "mr. zelner",
                "mr zellner": "mr. zelner",
                "mrs green": "mrs. greene",
                "mrs\. green": "mrs. greene",
                "dr\. green": "mr. greene",
                "dr\. leedbetter": "dr. ledbetter",
                "frank": "frank jr."}


friends_df['speaker'] = friends_df.speaker.apply(fix_names, args=(replacements,))
friends_df.groupby("speaker").size().reset_index(name="count").sort_values("count", ascending=False)[:100]
friends_df.groupby("speaker").size().reset_index(name="count").sort_values("speaker")
friends_df.speaker.nunique()

friends_df.to_csv("../data/friends/friends_lines_v2.csv", index=False, encoding="utf-8")

friends_df = pd.read_csv("../data/friends/friends_lines_v2.csv")
friends_df.groupby("scene").agg(speakers=("speaker",lambda x: x.tolist()))

######
friends_df.head()
print("Shape: ", friends_df.shape)

friends_df.groupby("speaker").size().reset_index(name="count").sort_values("count", ascending=False)
friends_df.speaker.nunique()   # 739

lines_by_season = friends_df.groupby('season')['line'].count()
lines_by_season.plot.bar(title="Number of lines by season", ylabel="Number of lines", rot=0)
plt.savefig("../figures/friends_lines_by_season.png")

scenes_by_season = friends_df.groupby('season')['scene'].nunique()
scenes_by_season.plot.bar(title="Number of scenes by season", ylabel="Number of scenes", rot=0)
plt.savefig("../figures/friends_scenes_by_season.png")

lines_by_speaker = friends_df.groupby(['speaker', 'season'])['line'].size().unstack(fill_value=0)
lines_by_speaker["sum_col"] = lines_by_speaker.sum(axis=1)
lines_by_speaker = lines_by_speaker.sort_values(by="sum_col", ascending=False)
lines_by_speaker = lines_by_speaker.drop("sum_col", axis=1)
lines_by_speaker[:15].plot(kind="bar", stacked=True, colormap="inferno", title="Lines spoken by character", ylabel="Number of lines", figsize=(12,8), rot=45)
plt.savefig("../figures/friends_speakers_by_season.png")

episodes_by_speaker = friends_df.loc[:, ['speaker', 'season', 'episode']].drop_duplicates().groupby(['speaker'])['speaker'].count().sort_values(ascending=False)[:15]
episodes_by_speaker.plot(kind="bar", title="Number of episodes in which characters occurred", ylabel="Number of episodes", figsize=(12,8), rot=45)
plt.show()

episodes_by_number_of_speaker = friends_df.loc[:, ['speaker', 'season', 'episode']].drop_duplicates().groupby(['season', 'episode'])['speaker'].count().sort_values(ascending=False)[:20]
episodes_by_number_of_speaker.plot(kind="bar", title="Number of characters in episodes", ylabel="Number of characters", figsize=(12,8), rot=45)
plt.show()

episodes_by_number_of_speaker = friends_df.loc[:, ['speaker', 'season', 'episode']].drop_duplicates().groupby(['season', 'episode'])['speaker'].count().sort_values(ascending=True)[:20]
episodes_by_number_of_speaker.plot(kind="bar", title="Number of characters in episodes", ylabel="Number of characters", figsize=(12,8), rot=45)
plt.show()