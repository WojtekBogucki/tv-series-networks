from utils import *
import os
import matplotlib

matplotlib.use('Agg')

path_dir = "../data/movies"
movie_paths = [os.path.join(path_dir, movie) for movie in os.listdir(path_dir)]
movie_titles = [movie.split(".")[0] for movie in os.listdir(path_dir)]

os.makedirs("../figures/movies", exist_ok=True)

i = 116
for movie_path, movie_title in zip(movie_paths[i:], movie_titles[i:]):
    print(movie_title)
    edges_weighted = pd.read_csv(movie_path)
    net = nx.from_pandas_edgelist(edges_weighted, source="speaker1", target="speaker2",
                                  edge_attr=["line_count", "scene_count", "word_count"])
    draw_interaction_network_communities(net, "line_count", method="LD", filename=f"movies/{movie_title}_lines")
    draw_interaction_network_communities(net, "scene_count", method="LD", filename=f"movies/{movie_title}_scenes")
    draw_interaction_network_communities(net, "word_count", method="LD", filename=f"movies/{movie_title}_words")