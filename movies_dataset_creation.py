import pandas as pd
import ast
import re
from processing.processing import get_valid_filename

if __name__ == "__main__":
    movies = pd.read_csv("data/movie_dialog_corpus/movie_titles_metadata.txt",
                         sep=r" \+\+\+\$\+\+\+ ",
                         header=None,
                         encoding="ISO-8859-2",
                         names=["id", "title", "year", "rating", "num_votes", "genres"],
                         engine="python")
    conversations = pd.read_csv("data/movie_dialog_corpus/movie_conversations.txt",
                                sep=r" \+\+\+\$\+\+\+ ",
                                header=None,
                                names=["speaker1", "speaker2", "movie", "lines"],
                                encoding="ISO-8859-2",
                                engine="python")
    characters = pd.read_csv("data/movie_dialog_corpus/movie_characters_metadata.txt",
                             sep=r" \+\+\+\$\+\+\+ ",
                             header=None,
                             names=["id", "speaker", "movieid", "title", "gender", "pos_in_credits"],
                             encoding="ISO-8859-2",
                             engine="python")
    lines = pd.read_csv("data/movie_dialog_corpus/movie_lines.txt",
                        sep=r" \+\+\+\$\+\+\+ ",
                        header=None,
                        names=["lineid", "characterid", "movieid", "character_name", "line"],
                        encoding="ISO-8859-2",
                        engine="python")

    # parse lists
    conversations["lines"] = conversations["lines"].apply(ast.literal_eval)
    con_lines = conversations.explode("lines").reset_index()

    # count words in lines
    lines["word_count"] = lines.apply(lambda x: len(re.split(r"[ ']", str(x.line))), axis=1)

    con_lines = con_lines.merge(lines[["lineid", "word_count"]], left_on="lines", right_on="lineid", how="left")
    con_lines.drop(["lines", "lineid"], axis=1, inplace=True)

    conv_scenes = con_lines.groupby(["index", "movie", "speaker1", "speaker2"]).agg(word_count=("word_count", "sum"),
                                                                                    line_count=(
                                                                                        "index", "count")).reset_index()

    movies_edges = conv_scenes.groupby(["movie", "speaker1", "speaker2"]).agg(line_count=("line_count", "sum"),
                                                                              scene_count=("index", "count"),
                                                                              word_count=(
                                                                                  "word_count", "sum")).reset_index()

    # get speaker's names and movie's titles
    movies_edges = movies_edges.merge(characters[["id", "speaker"]], left_on="speaker1", right_on="id", how="left").drop(
        ["id", "speaker1"], axis=1).rename(columns={"speaker": "speaker1"})
    movies_edges = movies_edges.merge(characters[["id", "speaker"]], left_on="speaker2", right_on="id", how="left").drop(
        ["id", "speaker2"], axis=1).rename(columns={"speaker": "speaker2"})
    movies_edges = movies_edges.merge(movies[["id", "title"]], left_on="movie", right_on="id", how="left").drop(
        ["id", "movie"], axis=1)

    # save each movie to csv file
    movie_titles = movies["title"].values.tolist()
    for title in movie_titles:
        movie_edges = movies_edges[movies_edges["title"] == title]
        title_name = get_valid_filename(title)
        movie_edges[["speaker1", "speaker2", "line_count", "scene_count", "word_count"]].to_csv(
            f"data/movies/{title_name}.csv", index=False, encoding="utf-8")

    # save ratings

    movies["genres"] = movies["genres"].apply(ast.literal_eval)

    movies_genres = movies.explode("genres")
    movies_genres = pd.get_dummies(movies_genres, columns=["genres"], prefix="", prefix_sep="").groupby(
        ["title", "id", "year", "rating", "num_votes"]).sum().reset_index()
    movies_genres["title"] = movies_genres["title"].apply(get_valid_filename)
    movies_genres.to_csv("data/imdb/movie_ratings.csv", index=False, encoding="utf-8")
