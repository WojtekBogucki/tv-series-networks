import pandas as pd
import matplotlib.pyplot as plt
from EDA.processing import fix_names

pd.options.display.max_columns = 10
pd.options.display.max_rows = None

seinfeld_df = pd.read_csv("../data/seinfeld/seinfeld_lines_v1.csv", encoding="utf-8")

seinfeld_df.head()
print("Shape: ", seinfeld_df.shape)

seinfeld_df.groupby("speaker").size().reset_index(name="count").sort_values("count", ascending=False)
seinfeld_df.speaker.nunique()   # 1060

lines_by_season = seinfeld_df.groupby('season')['line'].count()
lines_by_season.plot.bar(title="Number of lines by season", ylabel="Number of lines")
plt.xticks(rotation=0)
plt.savefig("../figures/seinfeld_lines_by_season.png")

scenes_by_season = seinfeld_df.groupby('season')['scene'].nunique()
scenes_by_season.plot.bar(title="Number of scenes by season", ylabel="Number of scenes")
plt.xticks(rotation=0)
plt.savefig("../figures/seinfeld_scenes_by_season.png")

lines_by_speaker = seinfeld_df.groupby(['speaker', 'season'])['line'].size().unstack(fill_value=0)
lines_by_speaker["sum_col"] = lines_by_speaker.sum(axis=1)
lines_by_speaker = lines_by_speaker.sort_values(by="sum_col", ascending=False)
lines_by_speaker = lines_by_speaker.drop("sum_col", axis=1)
lines_by_speaker[:15].plot(kind="bar", stacked=True, colormap="inferno", title="Lines spoken by character", ylabel="Number of lines", figsize=(12,8))
plt.xticks(rotation=45)
plt.savefig("../figures/seinfeld_speakers_by_season.png")

episodes_by_speaker = seinfeld_df.loc[:, ['speaker', 'season', 'episode']].drop_duplicates().groupby(['speaker'])['speaker'].count().sort_values(ascending=False)[:15]
episodes_by_speaker.plot(kind="bar", title="Number of episodes in which characters occurred", ylabel="Number of episodes", figsize=(12,8))
plt.xticks(rotation=45)
plt.show()

episodes_by_number_of_speaker = seinfeld_df.loc[:, ['speaker', 'season', 'episode']].drop_duplicates().groupby(['season', 'episode'])['speaker'].count().sort_values(ascending=False)[:20]
episodes_by_number_of_speaker.plot(kind="bar", title="Number of characters in episodes", ylabel="Number of characters", figsize=(12,8))
plt.xticks(rotation=45)
plt.show()


seinfeld_df.groupby("speaker").size().reset_index(name="count").sort_values("count", ascending=False)

replacements = {"pitt": "mr. pitt",
                "steinbrenner": "mr. steinbrenner",
                "lippman": "mr. lippman",
                "mrs. costanza": "estelle",
                "devola": "joe divola",
                "leo": "uncle leo"}


seinfeld_df['speaker'] = seinfeld_df.speaker.apply(fix_names, args=(replacements,))
seinfeld_df.groupby("speaker").size().reset_index(name="count").sort_values("count", ascending=False)[:100]
seinfeld_df.speaker.nunique()

seinfeld_df.to_csv("../data/seinfeld/seinfeld_lines_v2.csv", index=False, encoding="utf-8")