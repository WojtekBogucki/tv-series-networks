import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from EDA.processing import fix_names

pd.options.display.max_columns = 10
pd.options.display.max_rows = None

# read data
# office_raw = pd.read_csv("../data/the_office/The-Office-Lines-V4.csv", sep=",", quotechar='\"', skipinitialspace=True)

# fix unescaped commas
# office_raw.head()
# wrong_lines = office_raw.iloc[:, 6].notna()
# office_raw.loc[wrong_lines, 'line'] = office_raw.iloc[:, [5, 6]][wrong_lines].apply(lambda x: "{0}, {1}".format(x[0], x[1]), axis=1)
# office_raw.iloc[:, 0:6].to_csv("../data/the_office/the_office_lines_v5.csv", index=False, encoding="utf-8")

# read fixed data

office_raw = pd.read_csv("../data/the_office/the_office_lines_v5.csv")

office_raw.head()

lines_by_season = office_raw.groupby('season')['line'].count()
lines_by_season.plot.bar(title="Number of lines by season", ylabel="Number of lines", rot=0)
plt.savefig("../figures/office_lines_by_season.png")

lines_by_speaker = office_raw.groupby(['speaker', 'season'])['line'].size().unstack(fill_value=0)
lines_by_speaker["sum_col"] = lines_by_speaker.sum(axis=1)
lines_by_speaker = lines_by_speaker.sort_values(by="sum_col", ascending=False)
lines_by_speaker = lines_by_speaker.drop("sum_col", axis=1)
lines_by_speaker[:20].plot(kind="bar", stacked=True, colormap="inferno", title="Lines spoken by character", ylabel="Number of lines", figsize=(12, 8), rot=45)
plt.savefig("../figures/office_speakers_by_season.png")

episodes_by_speaker = office_raw.loc[:, ['speaker', 'season', 'episode']].drop_duplicates().groupby(['speaker'])['speaker'].count().sort_values()[-20:]
episodes_by_speaker.plot(kind="barh", rot=0)

episodes_by_number_of_speaker = office_raw.loc[:, ['speaker', 'season', 'episode']].drop_duplicates().groupby(['season', 'episode'])['speaker'].count().sort_values(ascending=False)[:20]
episodes_by_number_of_speaker.plot.bar()

replacements = {"Micheal": "Michael",
                "Todd": "Todd Packer",
                "Packer": "Todd Packer",
                "Robert": "Robert California",
                "Carroll": "Carol",
                "David": "David Wallace",
                "DeAngelo": "Deangelo",
                "Daryl": "Darryl",
                "Diane": "Diane Kelly",
                "A\.J\.": "AJ",
                "Bob": "Bob Vance",
                "Paul": "Paul Faust",
                "Holy": "Holly"}


office_raw['speaker'] = office_raw.speaker.apply(fix_names, args=(replacements,))
office_raw.groupby('speaker').size().reset_index(name="count").sort_values("speaker", ascending=False)
office_raw.to_csv("../data/the_office/the_office_lines_v6.csv", index=False, encoding="utf-8")

office_raw = pd.read_csv("../data/the_office/the_office_lines_v6.csv")
len(office_raw.speaker[office_raw.speaker.str.contains(" and ")])
len(office_raw.speaker[office_raw.speaker.str.contains(", ")])
len(office_raw.speaker[office_raw.speaker.str.contains("/")])
office_raw.speaker[office_raw.speaker.str.contains("/")]
len(office_raw.speaker[office_raw.speaker.str.contains(" & ")])

for splitter in [" and ", ", ", " & "]:
    filter_speakers = office_raw.speaker.str.contains(splitter)
    office_raw.loc[filter_speakers, "speaker"] = office_raw.speaker[filter_speakers].apply(lambda x: x.split(splitter))
    office_raw = office_raw.explode("speaker")

many_speakers = office_raw.speaker.str.contains(" and ")
office_raw.loc[many_speakers, "speaker"] = office_raw.speaker[many_speakers].apply(lambda x: x.split(" and "))

office_raw.to_csv("../data/the_office/the_office_lines_v7.csv", index=False, encoding="utf-8")