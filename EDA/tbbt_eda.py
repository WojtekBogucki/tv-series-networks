import pandas as pd
import matplotlib.pyplot as plt
from EDA.processing import fix_names, split_characters, remove_speakers, fix_filtered_names, distinguish_characters

pd.options.display.max_columns = 10
pd.options.display.max_rows = None

tbbt_df = pd.read_csv("../data/tbbt/tbbt_lines_v1.csv", encoding="utf-8")

tbbt_df.speaker = tbbt_df.speaker.apply(lambda x: x.replace(" together", ""))

len(tbbt_df.speaker[tbbt_df.speaker.str.contains(" and ")])
len(tbbt_df.speaker[tbbt_df.speaker.str.contains(" & ")])
tbbt_df = split_characters(tbbt_df, [" and ", " & "])

speakers_to_remove = ["1.  sheldon",
                      "1. amy",
                      "2.  amy",
                      "3.  amy",
                      "3.  sheldon",
                      "4. amy",
                      "4. sheldon",
                      "5.  sheldon",
                      "all",
                      "all three",
                      "announcement",
                      "answerphone",
                      "both",
                      "boys",
                      "caption",
                      "computer voice",
                      "dead mrs wolowitz",
                      "everybody",
                      "everyone",
                      "female voice",
                      "ghostly voice",
                      "gps",
                      "howard’s phone",
                      "iss voice",
                      "like this",
                      "man on tv",
                      "man on screen",
                      "man’s voice",
                      "mass",
                      "mechanical voice on sheldon’s phone",
                      "montage of scenes",
                      "phone",
                      "phone rings. answering machine",
                      "random voice",
                      "raj’s voice",
                      "secne",
                      "sheldon’s phone",
                      "sheldon’s voice",
                      "story",
                      "tech support voice",
                      "teleplay",
                      "television voice",
                      "together",
                      "various",
                      "various others",
                      "voice from buzzer",
                      "voice from outside",
                      "voice from television",
                      "voice from tv",
                      "voice inside",
                      "voice of spock",
                      "voice on television",
                      "voice on tv",
                      "woman’s voice",
                      "woman on tv"
                      ]

replacements = {"howard’s mother": "mrs wolowitz",
                "barry": "kripke",
                "barry kripke": "kripke",
                "lesley": "leslie",
                "leslie winkle": "leslie",
                "beverley": "beverly",
                "dr hofstadter": "beverly",
                "mrs hofstadter": "beverly",
                "mary": "mrs cooper",
                "mr\. rostenkowski": "mike",
                "mr rostenkowski": "mike",
                "mike r": "mike",
                "mike rostenkowski": "mike",
                "mike m": "mike massimino",
                "dr koothrapalli": "dr koothrappali",
                "wil wheaton": "wil",
                "gablehouser": "gablehauser",
                "dr gablehouser": "gablehauser",
                "seibert": "siebert",
                "dr\. seibert": "siebert",
                "hawking": "stephen hawking",
                "stephen": "stephen hawking",
                "ms jenson": "alex",
                "rajj": "raj",
                "bermadette": "bernadette",
                "mrs koothrapalli": "mrs koothrappali",
                "past sheldon": "sheldon",
                "past leonard": "leonard",
                "past howard": "howard",
                "past raj": "raj",
                "past penny": "penny",
                "col\. williams": "colonel williams",
                "col williams": "colonel williams",
                "col": "colonel williams",
                "adam west": "adam",
                "alfred hofstadter": "alfred",
                "bill": "bill nye",
                "brent spiner": "brent",
                "dr\. brian greene": "greene",
                "emly": "emily",
                "howatd": "howard",
                "ira flatow": "ira",
                "james earl jones": "james",
                "katee sackhoff": "katee",
                "kevn": "kevin",
                "leoanard": "leonard",
                "mary cooper": "mrs cooper",
                "nathan": "nathan fillion",
                "ramona nowitzki": "ramona",
                "rai": "raj",
                "sehldon": "sheldon",
                "sgeldon": "sheldon",
                "shedon": "sheldon",
                "shelldon": "sheldon",
                "shldon": "sheldon"}

tbbt_df = remove_speakers(tbbt_df, speakers_to_remove)
tbbt_df['speaker'] = tbbt_df.speaker.apply(fix_names, args=(replacements,))
tbbt_df = distinguish_characters(tbbt_df, ["girl", "guy", "man", "waiter", "waitress", "woman"])
tbbt_df.groupby("speaker").size().reset_index(name="count").sort_values("speaker", ascending=False)
tbbt_df.groupby("speaker").size().reset_index(name="count").sort_values("count", ascending=False)[:100]
tbbt_df.speaker.nunique()

tbbt_df.to_csv("../data/tbbt/tbbt_lines_v2.csv", index=False, encoding="utf-8")

tbbt_df = pd.read_csv("../data/tbbt/tbbt_lines_v2.csv")
tbbt_df.groupby(["season", "episode", "title"]).count()
################
tbbt_df.head()
print("Shape: ", tbbt_df.shape)

tbbt_df.groupby("speaker").size().reset_index(name="count").sort_values("count", ascending=False)
tbbt_df.speaker.nunique()   # 343

lines_by_season = tbbt_df.groupby('season')['line'].count()
lines_by_season.plot.bar(title="Number of lines by season", ylabel="Number of lines")
plt.xticks(rotation=0)
plt.savefig("../figures/tbbt_lines_by_season.png")

scenes_by_season = tbbt_df.groupby('season')['scene'].nunique()
scenes_by_season.plot.bar(title="Number of scenes by season", ylabel="Number of scenes")
plt.xticks(rotation=0)
plt.savefig("../figures/tbbt_scenes_by_season.png")

lines_by_speaker = tbbt_df.groupby(['speaker', 'season'])['line'].size().unstack(fill_value=0)
lines_by_speaker["sum_col"] = lines_by_speaker.sum(axis=1)
lines_by_speaker = lines_by_speaker.sort_values(by="sum_col", ascending=False)
lines_by_speaker = lines_by_speaker.drop("sum_col", axis=1)
lines_by_speaker[:15].plot(kind="bar", stacked=True, colormap="inferno", title="Lines spoken by character", ylabel="Number of lines", figsize=(12,8))
plt.xticks(rotation=45)
plt.savefig("../figures/tbbt_speakers_by_season.png")

episodes_by_speaker = tbbt_df.loc[:, ['speaker', 'season', 'episode']].drop_duplicates().groupby(['speaker'])['speaker'].count().sort_values(ascending=False)[:15]
episodes_by_speaker.plot(kind="bar", title="Number of episodes in which characters occurred", ylabel="Number of episodes", figsize=(12,8))
plt.xticks(rotation=45)
plt.show()

episodes_by_number_of_speaker = tbbt_df.loc[:, ['speaker', 'season', 'episode']].drop_duplicates().groupby(['season', 'episode'])['speaker'].count().sort_values(ascending=False)[:20]
episodes_by_number_of_speaker.plot(kind="bar", title="Number of characters in episodes", ylabel="Number of characters", figsize=(12,8))
plt.xticks(rotation=45)
plt.show()