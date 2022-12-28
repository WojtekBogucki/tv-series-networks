# tv-series-neworks
Repository for scripts used for Master's Thesis 

Title: *Analysis of graphs of the interactions between the characters in movies and TV series*

Author: Wojciech Bogucki

Supervisor: Grzegorz Siudem, PhD

University: Warsaw University of Technology

Faculty: Faculty of Mathematics and Information Sciences

## Project structure

### Web scrapers

Web scrapers for downloading transcriptions for Seinfeld, The Big Bang Theory and Friends.

Files are available in `web_scrapers` directory and are executed in script `run_scrapers.py`.

### Processing
Processing transcripts in tabular form - character unification.

Files are available in `processing` directory and are executed in script `run_processing.py`.

### Dataset creation

Creation of interactions dataset from transcripts in tabular form.

There is a script for TV series - `dataset_cration.py` and for movies - `movies_dataset_creation.py` .

### EDA
Performing EDA on TV series datasets and saving all plots in `figures` directory.

There is a script `run_eda.py` for executing it.

Created figures:
- top episodes by number of speakers
- top speakers by number of episodes they appeared in
- number of lines per season
- average number of lines in episode per season
- number of scenes by season
- boxplot of number of scenes per season
- average number of scenes in episode per season
- top speakers broken by number of lines per season

### Viewership and ratings

Processing ratings and viewership data for TV series.

Files are available in `viewership` and `ratings` directories and are executed in scripts `run_viewership.py` and `run_ratings.py`.

### Network analysis

Scripts used for all network analyses presented in master thesis are located in `networks` directory and should be run from this directory.

The following scripts are available:
* `network_analysis.py` - general analysis of TV series
* `community_detection_comparison.py` - comparison of community detection methods
* `feature_importance.py` - feature importance for model predicting ratings based on networks' structure
* `movie_networks.py` - comparison of networks of movies and TV series
* `network_clustering.py`- clustering of networks
* `office_seinfeld_comparison.py` - comparative analysis of Seinfeld and The Office