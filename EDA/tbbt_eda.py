import pandas as pd
from EDA.processing import fix_names, split_characters, remove_speakers, distinguish_characters, visualize_eda

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

# EDA
tbbt_df = pd.read_csv("../data/tbbt/tbbt_lines_v2.csv")
tbbt_df.groupby(["season", "episode", "title"]).count()
tbbt_df.head()
print("Shape: ", tbbt_df.shape)

tbbt_df.groupby("speaker").size().reset_index(name="count").sort_values("count", ascending=False)
tbbt_df.speaker.nunique()   # 351

visualize_eda(tbbt_df, "tbbt")
