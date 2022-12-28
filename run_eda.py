import logging
import os
import pandas as pd
from processing.processing import visualize_eda

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

if __name__ == "__main__":
    for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
        logger.info(f"Performing EDA for {show_name}...")
        path = f"data/{show_name}"

        # load raw data
        latest_file = [f for f in os.listdir(path) if f.startswith(f"{show_name}_lines_v")][-1]
        data = pd.read_csv(f"{path}/{latest_file}")

        logger.info(f"Shape: {data.shape}")
        logger.info(f"Unique speakers: {data.speaker.nunique()}")
        visualize_eda(data, show_name)