import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from EDA.processing import fix_names, split_characters, remove_speakers, fix_filtered_names, distinguish_characters

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

office_raw['speaker'] = office_raw.speaker.apply(fix_names, args=({"DunMiff\/sys": "DunMiffsys","Bob Vance, Vance Refrigeration": "Bob Vance"},))
office_raw = split_characters(office_raw, [" and ", ", ", " & ", "/"])

speakers_to_remove = ['"Phyllis"',
                      '"Jo"',
                      '"Jim"',
                      '"Angela"',
                      "Boom Box",
                      "Computron",
                      "song",
                      "Video",
                      "Various",
                      "Unknown",
                      "Together",
                      "TV",
                      "Song",
                      "Hunter's CD",
                      "Radio",
                      "Phone",
                      "Automated phone voice",
                      "Jim's voicemail",
                      "Ryan's Voicemail",
                      "Voicemail",
                      "Voice on CD player",
                      "others",
                      "Others",
                      "Oscar's Computer",
                      "Offscreen",
                      "Off-camera",
                      "GPS",
                      "Entire office",
                      "Office",
                      "Employees",
                      "Employees except Dwight",
                      "Everyone",
                      "Everyone watching",
                      "Everybody",
                      "Both",
                      "All",
                      "All Girls",
                      "All but Oscar",
                      "All the Men",
                      "Group",
                      "Teammates",
                      "Warehouse Crew",
                      "Crowd",
                      "Narrator",
                      "New Instant Message"]

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
                "Micahel": "Michael",
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
                "Julius Irving": "Julius",
                "Casey Dean": "Casey",
                "Carol Stills": "Carol",
                "Cousin Mose": "Mose",
                "Bar Manager": "Donna",
                "Fred Henry": "Fred",
                "Mee-Maw": "MeeMaw",
                "Mema": "MeeMaw",
                "Phil Maguire": "Phil",
                "Maguire": "Phil",
                "Philip": "Phillip",
                "Sensei Ira": "Ira",
                "Sensei": "Ira"
                }

office_raw = remove_speakers(office_raw, speakers_to_remove)

office_raw = fix_filtered_names(office_raw, [[7, 14]], {"David": "David Brent"})
office_raw = fix_filtered_names(office_raw, [[7, 18]], {"Holy": "Todd Packer"})
office_raw = fix_filtered_names(office_raw, [[5, 6], [6,4]], {"Pete": "Pete Halpert"})
office_raw = fix_filtered_names(office_raw, [[9, 19]], {"Carla": "Carla Fern"})
office_raw = fix_filtered_names(office_raw, [[2, 2]], {"Billy": "Billy Merchant"})
office_raw = fix_filtered_names(office_raw, [[2, 9]], {"Bill": "Bill from improv"})
office_raw = fix_filtered_names(office_raw, [[9, 16]], {"Mark": "Mark Franks"})
office_raw = fix_filtered_names(office_raw, [[5, 16]], {"Mark": "Mark Baldy"})


office_raw['speaker'] = office_raw.speaker.apply(fix_names, args=(replacements,))
office_raw = distinguish_characters(office_raw, ["Man", "Woman", "Guy", "Girl", "Waiter"])
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