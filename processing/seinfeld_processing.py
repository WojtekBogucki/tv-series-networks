import pandas as pd
from processing.processing import fix_names, split_characters, remove_speakers, fix_filtered_names, distinguish_characters
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def run_seinfeld_processing():
    logger.info("Started Seinfeld processing...")
    seinfeld_df = pd.read_csv("data/seinfeld/seinfeld_lines_v1.csv", encoding="utf-8")
    # len(seinfeld_df.speaker[seinfeld_df.speaker.str.contains(" and ")])
    # len(seinfeld_df.speaker[seinfeld_df.speaker.str.contains(" & ")])
    # len(seinfeld_df.speaker[seinfeld_df.speaker.str.contains("&")])
    # len(seinfeld_df.speaker[seinfeld_df.speaker.str.contains(", ")])

    seinfeld_df.groupby("speaker").size().reset_index(name="count").sort_values("speaker", ascending=False)
    seinfeld_df = split_characters(seinfeld_df, [", ", " and ", " & ", "&"])

    speakers_to_remove = ["all",
                          "all four men",
                          "all three",
                          "audience",
                          "both",
                          "building a",
                          "building b",
                          "building c",
                          "becoming board",
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
                          "together"
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
                                                       "man #1", "man #2", "man #3", "waiter", "waitress", "woman",
                                                       "worker"])
    # save data
    seinfeld_df.to_csv("data/seinfeld/seinfeld_lines_v2.csv", index=False, encoding="utf-8")
    logger.info("Finished Seinfeld processing.")

