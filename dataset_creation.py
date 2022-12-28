from processing.processing import *
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

if __name__ == "__main__":
    for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
        logger.info(f"Creating datasets for {show_name}...")
        path = f"data/{show_name}"

        # load raw data
        latest_file = [f for f in os.listdir(path) if f.startswith(f"{show_name}_lines_v")][-1]
        data = pd.read_csv(f"{path}/{latest_file}")

        # save all
        edges_weighted = (data.pipe(filter_by_speakers, count=0)
                              .pipe(filter_group_scenes)
                              .pipe(get_speaker_network_edges))
        edges_weighted.to_csv(f"{path}/edges_weighted_all.csv", index=False, encoding="utf-8")

        # save speakers with over 100 lines
        office_edges_weighted = (data.pipe(filter_by_speakers)
                                 .pipe(filter_group_scenes)
                                 .pipe(get_speaker_network_edges))
        office_edges_weighted.to_csv(f"{path}/edges_weighted.csv", index=False, encoding="utf-8")

        # top 30 speakers
        top30_edges_weighted = (data.pipe(filter_by_speakers, top=30)
                                .pipe(filter_group_scenes)
                                .pipe(get_speaker_network_edges))
        top30_edges_weighted.to_csv(f"{path}/edges_weighted_top30.csv", index=False, encoding="utf-8")

        save_seasons(data, count=20, path=path)
        save_episodes(data, count=0, path=path)

        save_merged_episodes(path)
        merge_seasons(path)
