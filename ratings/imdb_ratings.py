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

def merge_episodes(ratings, title, season, episodes):
    ep1_mask = (ratings.originalTitle == title) & (ratings.seasonNumber == season) & (ratings.episodeNumber == episodes[0])
    ep2_mask = (ratings.originalTitle == title) & (ratings.seasonNumber == season) & (ratings.episodeNumber == episodes[1])
    ep1 = ratings[ep1_mask]
    ep2 = ratings[ep2_mask]
    ep1_votes = ep1.numVotes.values
    ep2_votes = ep2.numVotes.values
    new_avg_rating = np.round(
        (ep1.averageRating.values * ep1_votes + ep2.averageRating.values * ep2_votes) / (ep1_votes + ep2_votes), 1)[0]
    ratings.loc[ep1_mask, "averageRating"] = new_avg_rating
    ratings.loc[ep1_mask, "numVotes"] = max(ep1_votes, ep2_votes)[0]
    ratings = ratings.drop(ep2.index)
    return ratings


ratings.loc[ratings.originalTitle=="The Big Bang Theory", "originalTitle"] = "tbbt"
ratings["originalTitle"] = ratings.originalTitle.apply(lambda x: x.lower().replace(" ", "_"))

ratings = merge_episodes(ratings, "the_office", 6, [4, 5])
ratings = merge_episodes(ratings, "the_office", 6, [17, 18])
ratings = merge_episodes(ratings, "seinfeld", 4, [1, 2])

# drop recap episodes
ratings.drop(ratings[(ratings.originalTitle=="seinfeld") & (ratings.seasonNumber==9) & (ratings.episodeNumber==21)].index, inplace=True)
ratings.drop(ratings[(ratings.originalTitle=="seinfeld") & (ratings.seasonNumber==6) & (ratings.episodeNumber==14)].index, inplace=True)

ratings.to_csv("../data/imdb/episode_ratings.csv", index=False, encoding="utf-8")

ratings = pd.read_csv("../data/imdb/episode_ratings.csv")


def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()


season_rating = ratings.groupby(["originalTitle", "seasonNumber"]).apply(w_avg, "averageRating", "numVotes").reset_index(name="weighted_rating")
season_rating.to_csv("../data/imdb/season_ratings.csv", index=False, encoding="utf-8")
