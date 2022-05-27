from utils import *
import os
import matplotlib

matplotlib.use('Agg')

path_dir = "../data/movies"
movie_paths = [os.path.join(path_dir, movie) for movie in os.listdir(path_dir)]
movie_titles = [movie.split(".")[0] for movie in os.listdir(path_dir)]

os.makedirs("../figures/movies", exist_ok=True)

i = 0
for movie_path, movie_title in zip(movie_paths[i:], movie_titles[i:]):
    print(movie_title)
    edges_weighted = pd.read_csv(movie_path)
    net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                                  edge_attr=["line_count", "scene_count", "word_count"])
    draw_interaction_network_communities(net, "line_count", method="LD", filename=f"movies/{movie_title}_lines")
    # draw_interaction_network_communities(net, "scene_count", method="LD", filename=f"movies/{movie_title}_scenes")
    # draw_interaction_network_communities(net, "word_count", method="LD", filename=f"movies/{movie_title}_words")

selected_movies = ["batman", "blade_runner", "braveheart", "citizen_kane", "dead_poets_society", "die_hard", "fargo",
                   "good_will_hunting", "hannibal", "independence_day", "jaws_2", "jurassic_park",
                   "monty_python_and_the_holy_grail", "pirates_of_the_caribbean", "saving_private_ryan", "scream",
                   "spider-man", "superman", "the_big_lebowski", "the_bourne_identity",
                   "the_godfather", "the_matrix", "titanic", "tomorrow_never_dies"]

movies_net = []
for movie_path, movie_title in zip(movie_paths, movie_titles):
    if movie_title in selected_movies:
        print(movie_title)
        edges_weighted = pd.read_csv(movie_path)
        net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                                      edge_attr=["line_count", "scene_count", "word_count"])
        movies_net.append(net)

movie_stats = get_movie_network_stats(movies_net, selected_movies)

create_similarity_matrix(movie_stats, selected_movies, filename="comparison/movies")
