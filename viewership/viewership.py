import pandas as pd
import numpy as np

pd.options.display.max_columns = 10

office_view = pd.read_csv("../data/viewership/the_office.csv")
office_view["episode"] = [item for n in office_view.groupby("Season").size().values.tolist() for item in np.arange(1,n+1)]
office_view = office_view[["Season", "episode", "EpisodeTitle", "Viewership"]]
office_view["show"] = "the_office"
office_view.columns = ["season", "episode", "title", "viewership", "show"]

seinfeld_view = pd.read_csv("../data/viewership/seinfeld.csv")
seinfeld_view["us_viewers"] = seinfeld_view["us_viewers"].apply(lambda x: np.round(x/1e6,1))
seinfeld_view = seinfeld_view[["season", "episode_num_in_season", "title", "us_viewers"]]
seinfeld_view["show"] = "seinfeld"
seinfeld_view.columns = ["season", "episode", "title", "viewership", "show"]

tbbt_view = pd.read_csv("../data/viewership/tbbt.csv")
tbbt_view = tbbt_view[["Season", "No. inseason", "Title", "U.S. viewers(millions)"]]
tbbt_view["show"] = "tbbt"
tbbt_view.columns = ["season", "episode", "title", "viewership", "show"]

friends_view = pd.read_csv("../data/viewership/friends.csv")
friends_view = friends_view.drop(friends_view[friends_view.Episode=="Special"].index)
friends_view["season"] = friends_view.Episode.apply(lambda x: int(x.split("-")[0]))
friends_view["episode"] = friends_view.Episode.apply(lambda x: int(x.split("-")[1][:2]))
friends_view["U.S. viewers"] = friends_view["U.S. viewers"].apply(lambda x: float(x.replace(" million", "")))
friends_view = friends_view[["season", "episode", "Title", "U.S. viewers"]]
friends_view["show"] = "friends"
friends_view.columns = ["season", "episode", "title", "viewership", "show"]

viewership = pd.concat([office_view, seinfeld_view, tbbt_view, friends_view], axis=0)
viewership.to_csv("../data/viewership/viewership.csv", index=False, encoding="utf-8")