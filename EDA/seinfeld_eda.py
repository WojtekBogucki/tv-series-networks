import pandas as pd
import matplotlib.pyplot as plt
from EDA.processing import fix_names, split_characters, remove_speakers, fix_filtered_names, distinguish_characters

pd.options.display.max_columns = 10
pd.options.display.max_rows = None

seinfeld_df = pd.read_csv("../data/seinfeld/seinfeld_lines_v1.csv", encoding="utf-8")

seinfeld_df.head()
print("Shape: ", seinfeld_df.shape)

len(seinfeld_df.speaker[seinfeld_df.speaker.str.contains(" and ")])
len(seinfeld_df.speaker[seinfeld_df.speaker.str.contains(" & ")])
len(seinfeld_df.speaker[seinfeld_df.speaker.str.contains("&")])
seinfeld_df[seinfeld_df.speaker.apply(len) > 20]

seinfeld_df.groupby("speaker").size().reset_index(name="count").sort_values("speaker", ascending=False)

seinfeld_df = split_characters(seinfeld_df, [" and "," & ", "&"])

speakers_to_remove = ["all",
                      "all four men",
                      "all three",
                      "audience",
                      "both",
                      "building a",
                      "building b",
                      "building c",
                      "clerk's voice",
                      "computer voice",
                      "crowd",
                      "dj on radio",
                      "elaine thinking",
                      "elaine's brain",
                      "elaine's voice",
                      "everybody",
                      "everyone",
                      "george's voice",
                      "group",
                      "high pitched voice",
                      "hospital voiceover",
                      "jerry's answering machine",
                      "jerry's outgoing message",
                      "jerry's stand-up",
                      "montage",
                      "off stage",
                      "opening monolog",
                      "opening scene",
                      "radio announcer",
                      "sam's voice",
                      "so far",
                      "tv",
                      "tv announcer",
                      "tv newscaster",
                      "tv voice",
                      "tvvoice",
                      "voice",
                      "voice 1",
                      "voice 2",
                      "voice from poker game",
                      "voice on intercom",
                      "voice on speaker",
                      "voice on the phone",
                      "woman's voice on the phone"
                      ]

replacements = {"pitt": "mr. pitt",
                "steinbrenner": "mr. steinbrenner",
                "lippman": "mr. lippman",
                "mrs. costanza": "estelle",
                "devola": "joe divola",
                "davola": "joe divola",
                "leo": "uncle leo",
                "mr ross": "mr. ross",
                "j\. peterman": "peterman",
                "mr. peterman": "peterman",
                "claie": "claire",
                "marry": "mary",
                "allsion": "allison",
                "mr\.thomassoulo": "thomassoulo",
                "mr\. thomassoulo": "thomassoulo",
                "babu bhatt": "babu",
                "docter": "doctor",
                "dry cleane": "dry cleaner",
                "elainelaine": "elaine",
                "eliane": "elaine",
                "goerge": "george",
                "izzy izzy sr\.\.": "izzy sr.",
                "izzy": "izzy sr.",
                "jerr": "jerry",
                "jery": "jerry",
                "krmaer": "kramer",
                "man 1": "man #1",
                "man 2": "man #2",
                "man 3": "man #3",
                "man#1": "man #1",
                "man#2": "man #2",
                "man#3": "man #3",
                "man1": "man #1",
                "man2": "man #2",
                "man3": "man #3",
                "marcellino": "marcelino",
                "newmanewman": "newman",
                "whatley": "tim"
                }

seinfeld_df = remove_speakers(seinfeld_df, speakers_to_remove)

seinfeld_df = fix_filtered_names(seinfeld_df, [[4, 17], [4, 21], [4, 24]], {"allison": "allison (season 4)"})
seinfeld_df = fix_filtered_names(seinfeld_df, [[8, 15]], {"allison": "allison (season 8)"})
seinfeld_df = fix_filtered_names(seinfeld_df, [[9, 2]], {"allison": "allison from play now"})
seinfeld_df = fix_filtered_names(seinfeld_df, [[7, 18]], {"bob": "bob grossberg"})
seinfeld_df = fix_filtered_names(seinfeld_df, [[5, 15]], {"bob": "bob from suit store"})
seinfeld_df = fix_filtered_names(seinfeld_df, [[8, 19]], {"father": "father curtis"})

seinfeld_df['speaker'] = seinfeld_df.speaker.apply(fix_names, args=(replacements,))
seinfeld_df = distinguish_characters(seinfeld_df, ["cashier", "cop", "doctor", "doorman", "guy", "judge", "kid", "man",
                                                   "man #1", "man #2", "man #3", "waiter", "waitress", "woman", "worker"])
seinfeld_df.groupby("speaker").size().reset_index(name="count").sort_values("count", ascending=False)[:100]
seinfeld_df.speaker.nunique()

seinfeld_df.to_csv("../data/seinfeld/seinfeld_lines_v2.csv", index=False, encoding="utf-8")

seinfeld_df = pd.read_csv("../data/seinfeld/seinfeld_lines_v2.csv")
seinfeld_df.groupby(["season", "episode", "title"]).count()

seinfeld_df.groupby("speaker").size().reset_index(name="count").sort_values("count", ascending=False)
seinfeld_df.speaker.nunique()   # 1116

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