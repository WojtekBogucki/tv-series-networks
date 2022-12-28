import pandas as pd
from processing.processing import fix_names, split_characters, remove_speakers, fix_filtered_names, distinguish_characters
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def run_friends_processing():
    logger.info("Started Friends processing...")
    friends_df = pd.read_csv("data/friends/friends_lines_v1.csv", encoding="utf-8")
    # len(friends_df.speaker[friends_df.speaker.str.contains(" and ")])
    # len(friends_df.speaker[friends_df.speaker.str.contains(", ")])
    # len(friends_df.speaker[friends_df.speaker.str.contains(" & ")])
    # len(friends_df.speaker[friends_df.speaker.str.contains(", and ")])

    friends_df = split_characters(friends_df, [", and ", " and ", ", ", " & "])
    # friends_df.groupby("speaker").size().reset_index(name="count").sort_values("speaker", ascending=False)

    speakers_to_remove = ["aired",
                          "all",
                          "both",
                          "boys",
                          "but she sees ross",
                          "commercial",
                          "commercial voiceover",
                          "cut to",
                          "directed by",
                          "dr. drake ramoray",
                          "dr. drake remoray",
                          "dream joey",
                          "dream monica",
                          "everybody",
                          "everyone",
                          "everyone almost simultaneously except ross",
                          "everyone but monica",
                          "gang",
                          "girls",
                          "guys",
                          "her friends",
                          "her friend",
                          "hold voice",
                          "intercom",
                          "joey on tv",
                          "man on tv",
                          "man's voice",
                          "narrator",
                          "note",
                          "others",
                          "priest on tv",
                          "radio",
                          "same man's voice",
                          "soothing male voice",
                          "teleplay",
                          "the guys",
                          "the girls",
                          "together",
                          "tv",
                          "tv announcer",
                          "tv doctor",
                          "voice",
                          "video",
                          "woman on tv",
                          "woman's voice"
                          ]

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
                    "dr green": "mr. greene",
                    "dr\. leedbetter": "dr. ledbetter",
                    "frank": "frank jr.",
                    "amger": "amber",
                    "billy crystal": "billy",
                    "c\.h\.e\.e\.s\.e": "c.h.e.e.s.e.",
                    "chander,": "chandler",
                    "chandler,": "chandler",
                    "chandlers": "chandler",
                    "dr horton": "dr. horton",
                    "gunter": "gunther",
                    "joey's grandmother": "grandma tribbiani",
                    "maitre d": "maitre d'",
                    "matire'd": "maitre d'",
                    "mike's dad": "mike's father",
                    "mike's mom": "mike's mother",
                    "monica,": "monica",
                    "phoebe sr\.": "phoebe sr",
                    "rache": "rachel",
                    "rachel,": "rachel",
                    "ross,": "ross"}

    friends_df = remove_speakers(friends_df, speakers_to_remove)

    friends_df = fix_filtered_names(friends_df, [[8, 10]], {"bobby": "bobby corso"})
    friends_df = fix_filtered_names(friends_df, [[8, 5]], {"bob": "bob (chandlers coworker)"})
    friends_df = fix_filtered_names(friends_df, [[5, 11]], {"elizabeth": "elizabeth hornswoggle"})
    friends_df = fix_filtered_names(friends_df, [[7, 19]], {"the casting director": "leslie (casting director)"})

    friends_df['speaker'] = friends_df.speaker.apply(fix_names, args=(replacements,))
    friends_df = distinguish_characters(friends_df, ["bob", "girl", "guy", "kid", "man", "the waiter", "the woman",
                                                     "the teacher", "the salesman", "waiter", "waitress", "woman"])
    # save data
    friends_df.to_csv("data/friends/friends_lines_v2.csv", index=False, encoding="utf-8")
    logger.info("Finished Friends processing.")

