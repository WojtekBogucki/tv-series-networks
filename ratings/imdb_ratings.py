import pandas as pd
import numpy as np

pd.options.display.max_columns = 10

basics = pd.read_csv("../data/imdb/basics.tsv", sep="\t", low_memory=False)

friends_filter = (basics.originalTitle=="Friends") & (basics.titleType=="tvSeries") & (basics.startYear=="1994")
seinfeld_filter = (basics.originalTitle=="Seinfeld") & (basics.titleType=="tvSeries")
office_filter = (basics.originalTitle=="The Office") & (basics.titleType=="tvSeries") & (basics.startYear=="2005")
tbbt_filter = (basics.originalTitle=="The Big Bang Theory") & (basics.titleType=="tvSeries")
basics = basics[(friends_filter) | (seinfeld_filter) | (office_filter) | (tbbt_filter)]

episode = pd.read_csv("../data/imdb/episode.tsv", sep="\t", low_memory=False)

episode = basics.merge(episode, left_on="tconst", right_on="parentTconst", how="left")

ratings = pd.read_csv("../data/imdb/ratings.tsv", sep="\t", low_memory=False)

ratings = episode.merge(ratings, left_on="tconst_y", right_on="tconst", how="left")

ratings = ratings.drop(["tconst_x", "tconst_y", "tconst", "parentTconst", "titleType", "primaryTitle", "isAdult", "runtimeMinutes"], axis=1)
ratings["episodeNumber"] = ratings.episodeNumber.astype("int")
ratings["seasonNumber"] = ratings.seasonNumber.astype("int")
ratings = ratings.sort_values(["originalTitle", "seasonNumber", "episodeNumber"])

ratings.to_csv("../data/imdb/episode_ratings.csv", index=False, encoding="utf-8")

ratings = pd.read_csv("../data/imdb/episode_ratings.csv")


def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()


season_rating = ratings.groupby(["originalTitle", "seasonNumber"]).apply(w_avg, "averageRating", "numVotes").reset_index(name="weighted_rating")
season_rating.to_csv("../data/imdb/season_ratings.csv", index=False, encoding="utf-8")
