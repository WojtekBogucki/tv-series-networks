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

office_s6_e4 = ratings[(ratings.originalTitle=="The Office") & (ratings.seasonNumber==6) & (ratings.episodeNumber==4)]
office_s6_e5 = ratings[(ratings.originalTitle=="The Office") & (ratings.seasonNumber==6) & (ratings.episodeNumber==5)]
s6e4_votes = office_s6_e4.numVotes.values
s6e5_votes = office_s6_e5.numVotes.values
new_avg_rating = np.round((office_s6_e4.averageRating.values*s6e4_votes + office_s6_e5.averageRating.values*s6e5_votes)/(s6e4_votes + s6e5_votes), 1)[0]
ratings.loc[(ratings.originalTitle=="The Office") & (ratings.seasonNumber==6) & (ratings.episodeNumber==4), "averageRating"] = new_avg_rating
ratings.loc[(ratings.originalTitle=="The Office") & (ratings.seasonNumber==6) & (ratings.episodeNumber==4), "numVotes"] = max(s6e4_votes, s6e5_votes)[0]
ratings.drop(office_s6_e5.index, inplace=True)

office_s6_e17 = ratings[(ratings.originalTitle=="The Office") & (ratings.seasonNumber==6) & (ratings.episodeNumber==17)]
office_s6_e18 = ratings[(ratings.originalTitle=="The Office") & (ratings.seasonNumber==6) & (ratings.episodeNumber==18)]
s6e17_votes = office_s6_e17.numVotes.values
s6e18_votes = office_s6_e18.numVotes.values
new_avg_rating = np.round((office_s6_e17.averageRating.values*s6e17_votes + office_s6_e18.averageRating.values*s6e18_votes)/(s6e17_votes + s6e18_votes), 1)[0]
ratings.loc[(ratings.originalTitle=="The Office") & (ratings.seasonNumber==6) & (ratings.episodeNumber==17), "averageRating"] = new_avg_rating
ratings.loc[(ratings.originalTitle=="The Office") & (ratings.seasonNumber==6) & (ratings.episodeNumber==17), "numVotes"] = max(s6e17_votes, s6e18_votes)[0]
ratings.drop(office_s6_e18.index, inplace=True)

ratings.to_csv("../data/imdb/episode_ratings.csv", index=False, encoding="utf-8")

ratings = pd.read_csv("../data/imdb/episode_ratings.csv")


def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()


season_rating = ratings.groupby(["originalTitle", "seasonNumber"]).apply(w_avg, "averageRating", "numVotes").reset_index(name="weighted_rating")
season_rating.to_csv("../data/imdb/season_ratings.csv", index=False, encoding="utf-8")
