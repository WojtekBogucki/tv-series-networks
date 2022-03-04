import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from EDA.processing import fix_names, split_characters

pd.options.display.max_columns = 10
pd.options.display.max_rows = None

# read data
office_raw = pd.read_csv("../data/the_office/The-Office-Lines-V4.csv", sep=",", quotechar='\"', skipinitialspace=True)

# fix unescaped commas
wrong_lines = office_raw.iloc[:, 6].notna()
office_raw.loc[wrong_lines, 'line'] = office_raw.iloc[:, [5, 6]][wrong_lines].apply(lambda x: f"{x[0]}, {x[1]}", axis=1)
office_raw.iloc[:, 0:6].to_csv("../data/the_office/the_office_lines_v5.csv", index=False, encoding="utf-8")

# read fixed data

office_raw = pd.read_csv("../data/the_office/the_office_lines_v5.csv")

office_raw = split_characters(office_raw, [" and ", ", ", " & ", "/"])

replacements = {"A\.J\.": "AJ",
                "Anglea": "Angela",
                "Angel": "Angela",
                "Angels": "Angela",
                "Bob": "Bob Vance",
                "Carroll": "Carol",
                "Carrol": "Carol",
                "Micheal": "Michael",
                "Michel": "Michael",
                "M ichael": "Michael",
                "MIchael": "Michael",
                "Michae": "Michael",
                "Micael": "Michael",
                "Michal": "Michael",
                "Miichael": "Michael",
                "Mihael": "Michael",
                "Todd": "Todd Packer",
                "Packer": "Todd Packer",
                "Robert": "Robert California",
                "Dacvid Wallace": "David Wallace",
                "David": "David Wallace",
                "Dacvid Walalce": "David Wallace",
                "David Wallcve": "David Wallace",
                "DeAngelo": "Deangelo",
                "Denagelo": "Deangelo",
                "DeAgnelo": "Deangelo",
                "Daryl": "Darryl",
                "Darry": "Darryl",
                "Diane": "Diane Kelly",
                "Paul": "Paul Faust",
                "Holy": "Holly",
                "Holly,": "Holly",
                "Marie": "Concierge Marie",
                "Concierge": "Concierge Marie",
                "DunMiff\/sys": "DunMiffsys",
                "JIm": "Jim",
                "JIM9334": "Jim",
                "Receptionitis15": "Pam",
                "sAndy": "Andy",
                "Stanely": "Stanley",
                "Dight": "Dwight",
                "D": "Dwight",
                "VRG": "Vance Refrigeration Guy",
                "VRG 1": "Vance Refrigeration Guy 1",
                "VRG 2": "Vance Refrigeration Guy 2",
                "Meridith": "Meredith",
                "Phyliss": "Phyllis",
                "abe": "Gabe",
                "Warren Buffett": "Warren",
                "Nellie Bertram": "Nellie",
                "Senator Liptop": "Senator Lipton",
                "Senator": "Senator Lipton",
                "\(Pam's mom\) Heleen": "Helene",
                "Pam's Mom": "Helene",
                "Pam's mom": "Helene",
                "Helen": "Helene",
                "Walt Jr\.": "Walter Jr",
                "Tom Halpert": "Tom",
                "Teddy Wallace": "Teddy",
                "Rolph": "Rolf",
                "Julius Irving": "Julius"
                }

filter_s07e14 = (office_raw.season == 7) & (office_raw.episode == 14)
office_raw.loc[filter_s07e14, 'speaker'] = office_raw.loc[filter_s07e14, 'speaker'].apply(fix_names, args=({"David": "David Brent"},))
filter_s07e18 = (office_raw.season == 7) & (office_raw.episode == 18)
office_raw.loc[filter_s07e18, 'speaker'] = office_raw.loc[filter_s07e18, 'speaker'].apply(fix_names, args=({"Holy": "Todd Packer"},))
filter_pete = ((office_raw.season == 5) & (office_raw.episode == 6)) | ((office_raw.season == 6) & (office_raw.episode == 4))
office_raw.loc[filter_pete, 'speaker'] = office_raw.loc[filter_pete, 'speaker'].apply(fix_names, args=({"Pete": "Pete Halpert"},))


office_raw['speaker'] = office_raw.speaker.apply(fix_names, args=(replacements,))
office_raw.groupby('speaker').size().reset_index(name="count").sort_values("speaker", ascending=False)
office_raw.to_csv("../data/the_office/the_office_lines_v6.csv", index=False, encoding="utf-8")

office_raw = pd.read_csv("../data/the_office/the_office_lines_v6.csv")
len(office_raw.speaker[office_raw.speaker.str.contains(" and ")])
len(office_raw.speaker[office_raw.speaker.str.contains(", ")])
len(office_raw.speaker[office_raw.speaker.str.contains("/")])
len(office_raw.speaker[office_raw.speaker.str.contains(" & ")])


###########
line_count = office_raw.groupby(["season", "speaker"]).size().reset_index(name="line_count")
line_count.loc[(line_count.season==2) & (line_count.line_count<100),["speaker", "line_count"]].plot(kind="hist", bins=12)
plt.ion()

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