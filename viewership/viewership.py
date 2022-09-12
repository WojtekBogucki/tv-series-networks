import pandas as pd
import numpy as np


def merge_episodes(df, show, season, episodes):
    ep1_mask = (df.show == show) & (df.season == season) & (df.episode == episodes[0])
    ep2_mask = (df.show == show) & (df.season == season) & (df.episode == episodes[1])
    ep1 = df[ep1_mask]
    ep2 = df[ep2_mask]
    ep1_view = ep1.viewership.values
    ep2_view = ep2.viewership.values
    df.loc[ep1_mask, "viewership"] = max(ep1_view, ep2_view)[0]
    df = df.drop(ep2.index)
    return df


def save_viewerships():
    office_view = pd.read_csv("data/viewership/the_office.csv")
    office_view["episode"] = [item for n in office_view.groupby("Season").size().values.tolist() for item in np.arange(1,n+1)]
    office_view = office_view[["Season", "episode", "EpisodeTitle", "Viewership"]]
    office_view["show"] = "the_office"
    office_view.columns = ["season", "episode", "title", "viewership", "show"]

    seinfeld_view = pd.read_csv("data/viewership/seinfeld.csv")
    seinfeld_view["us_viewers"] = seinfeld_view["us_viewers"].apply(lambda x: np.round(x/1e6,1))
    seinfeld_view = seinfeld_view[["season", "episode_num_in_season", "title", "us_viewers"]]
    seinfeld_view["show"] = "seinfeld"
    seinfeld_view.columns = ["season", "episode", "title", "viewership", "show"]

    tbbt_view = pd.read_csv("data/viewership/tbbt.csv")
    tbbt_view = tbbt_view[["Season", "No. inseason", "Title", "U.S. viewers(millions)"]]
    tbbt_view["show"] = "tbbt"
    tbbt_view.columns = ["season", "episode", "title", "viewership", "show"]

    friends_view = pd.read_csv("data/viewership/friends.csv")
    friends_view = friends_view.drop(friends_view[friends_view.Episode=="Special"].index)
    friends_view["season"] = friends_view.Episode.apply(lambda x: int(x.split("-")[0]))
    friends_view["episode"] = friends_view.Episode.apply(lambda x: int(x.split("-")[1][:2]))
    friends_view["U.S. viewers"] = friends_view["U.S. viewers"].apply(lambda x: float(x.replace(" million", "")))
    friends_view = friends_view[["season", "episode", "Title", "U.S. viewers"]]
    friends_view["show"] = "friends"
    friends_view.columns = ["season", "episode", "title", "viewership", "show"]

    viewership = pd.concat([office_view, seinfeld_view, tbbt_view, friends_view], axis=0).reset_index(drop=True)

    viewership = merge_episodes(viewership, "the_office", 6, [4, 5])
    viewership = merge_episodes(viewership, "the_office", 6, [17, 18])
    viewership = merge_episodes(viewership, "seinfeld", 3, [17, 18])
    viewership = merge_episodes(viewership, "seinfeld", 4, [1, 2])
    viewership = merge_episodes(viewership, "seinfeld", 4, [23, 24])
    viewership = merge_episodes(viewership, "seinfeld", 5, [18, 19])
    viewership = merge_episodes(viewership, "seinfeld", 7, [14, 15])
    viewership = merge_episodes(viewership, "seinfeld", 7, [21, 22])
    viewership = merge_episodes(viewership, "friends", 2, [12, 13])
    viewership = merge_episodes(viewership, "friends", 4, [23, 24])
    viewership = merge_episodes(viewership, "friends", 5, [23, 24])
    viewership = merge_episodes(viewership, "friends", 6, [15, 16])
    viewership = merge_episodes(viewership, "friends", 6, [24, 25])
    viewership = merge_episodes(viewership, "friends", 7, [23, 24])
    viewership = merge_episodes(viewership, "friends", 8, [23, 24])
    viewership = merge_episodes(viewership, "friends", 9, [23, 24])

    viewership.drop(viewership[(viewership.show == "seinfeld") & (viewership.season == 9) & (viewership.episode == 21)].index, inplace=True)
    viewership.drop(viewership[(viewership.show == "seinfeld") & (viewership.season == 9) & (viewership.episode == 22)].index, inplace=True)
    viewership.drop(viewership[(viewership.show == "seinfeld") & (viewership.season == 9) & (viewership.episode == 24)].index, inplace=True)
    viewership.drop(viewership[(viewership.show == "seinfeld") & (viewership.season == 6) & (viewership.episode == 14)].index, inplace=True)
    viewership.drop(viewership[(viewership.show == "seinfeld") & (viewership.season == 6) & (viewership.episode == 15)].index, inplace=True)
    viewership.drop(viewership[(viewership.show == "tbbt") & ((viewership.season == 11) | (viewership.season == 12))].index, inplace=True)

    viewership.to_csv("data/viewership/viewership.csv", index=False, encoding="utf-8")

    season_viewership = viewership.groupby(["show", "season"])["viewership"].mean().reset_index(name="avg_viewership")
    season_viewership.avg_viewership = season_viewership.avg_viewership.round(2)
    season_viewership.to_csv("data/viewership/season_viewership.csv", index=False, encoding="utf-8")