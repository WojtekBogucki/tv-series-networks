import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

main_URL = "https://bigbangtrans.wordpress.com/"
headers = {"User-Agent": "'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'"}


def get_episode_list(url: str, headers: dict) -> list:
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")
    pages = soup.find_all("div", {"id": "pages-2"})[0]
    elements = pages.findChildren("a")[1:]
    logger.info("Got episode list")
    return elements


def get_episode_titles(episode_list: list) -> list:
    episode_titles = list(map(lambda x: x.text
                    .strip()
                    .lower()
                    .replace("\xa0", " ")
                    .replace("series ", "s")
                    .replace(" episode ", "e")
                    .translate(str.maketrans({'\n': '', '–': '', ' ': '_', '(': '', ')': '', '/': ''}))
                    .replace("__", "_"),
                    episode_list))
    logger.info(f"Got {len(episode_titles)} episode titles")
    return episode_titles


def get_episode_links(episode_list: list) -> list:
    episode_links = list(map(lambda x: x.get("href").strip(), episode_list))
    logger.info(f"Got {len(episode_links)} episode links")
    return episode_links


def save_raw_scripts(episode_titles: list, episode_links: list, headers: dict, i: int = 0) -> None:
    for ep_title, ep_link in zip(episode_titles[i:], episode_links[i:]):
        i += 1
        logger.info(f"Episode {i}, {ep_title}")
        sub_page = requests.get(ep_link, headers=headers)
        soup_subpage = BeautifulSoup(sub_page.content, "html.parser")
        script_text = soup_subpage.find_all("div", {"class": "entrytext"})[0].get_text()
        script_text = re.sub(r'\n{2,}', '\n', script_text)
        script_text = re.sub(r'\nShare this:.*$', '', script_text)
        script_text = re.sub(r'\nWritten by.*$', '', script_text)
        script_text = re.sub(r'\nCredits sequence\.', '', script_text, re.IGNORECASE)
        try:
            with open(f"data/tbbt/raw_scripts/{ep_title}.txt", "w", encoding="utf-8") as f:
                f.write(script_text)
        except UnicodeEncodeError as uee:
            logger.error(uee)
    logger.info("Finished saving raw scripts")


def create_transcript_file(episode_titles: list) -> pd.DataFrame:
    entry_words = ["enters?", "entering", "walks? in", "approaches", "arriving", "opening door"]
    exit_words = ["exits?", "exiting", "leaves?", "walks? out", "hungs up", "flashback"]
    new_scene_words = entry_words + exit_words
    next_scene = False
    seasons = []
    episodes = []
    titles = []
    scenes = []
    speakers = []
    lines = []
    scene = 0
    for ep_title in episode_titles:
        pattern = re.compile(r'^s(\d{2})e(\d{2})_([a-z0-9_-]+)', re.IGNORECASE)
        info = pattern.search(ep_title)
        season = int(info.group(1))
        episode = int(info.group(2))
        title = info.group(3)
        if season == 2 and episode == 17:
            ep_title = "fixed/" + ep_title
        logger.info(f"Season {season}, episode {episode}, {title}")
        with open(f"data/tbbt/raw_scripts/{ep_title}.txt", "r", encoding="utf-8") as f:
            for line in f:
                full_line = line
                line = line.strip()
                if next_scene:
                    scene += 1
                    next_scene = False
                stage_dirs = re.findall(r"(\([^)]*\))", line)
                if stage_dirs:
                    for word in new_scene_words:
                        match = re.findall(fr"(\(.*{word}.*\))", line, re.IGNORECASE)
                        if match:
                            if word in entry_words or (word in exit_words and line.startswith("(")):
                                scene += 1
                                break
                            elif word in exit_words and not line.startswith("("):
                                next_scene = True
                line = re.sub(r"(\([^)]*\))", "", line)
                lower_line = line.lower()
                if not line:
                    continue
                elif lower_line.startswith("scene:") or \
                        lower_line.startswith("fantasy sequence") or \
                        lower_line.startswith("end fantasy") or \
                        lower_line.startswith("back to apartment") or \
                        lower_line.startswith("flash") or \
                        lower_line.startswith("time") or \
                        lower_line.startswith("shortly afterwards") or \
                        lower_line.startswith("cut to") or \
                        lower_line.startswith("howard’s car") or \
                        lower_line.startswith("leonards’s car") or \
                        lower_line.startswith("slight time shift") or \
                        lower_line.startswith("(time shift") or \
                        lower_line.startswith("(later") or \
                        lower_line.startswith("(back") or \
                        lower_line.startswith("later"):
                    scene += 1
                    continue
                elif lower_line.startswith("(") or lower_line.startswith(".") or lower_line.startswith("credit"):
                    continue
                pattern = re.compile(r"(^[A-Za-z0-9'.#& \"’\-]+): ? ?(.*)")
                line_search = pattern.search(line)
                if line_search is not None:
                    speaker = line_search.group(1)
                    line = line_search.group(2)
                else:
                    with open(f"data/tbbt/errors.txt", "a") as err:
                        err.write(f"{season} {episode} Line: {line}\n")
                    # print("*error*", line)
                    continue
                seasons.append(season)
                episodes.append(episode)
                titles.append(title)
                scenes.append(scene)
                speakers.append(speaker.strip().lower())
                lines.append(line.strip())
    tbbt_df = pd.DataFrame.from_dict({"season": seasons,
                                          "episode": episodes,
                                          "title": titles,
                                          "scene": scenes,
                                          "speaker": speakers,
                                          "line": lines})
    logger.info("Finished creating transcript file")
    return tbbt_df


def run_tbbt_scrapper():
    episode_list = get_episode_list(main_URL, headers)
    episode_titles = get_episode_titles(episode_list)
    episode_links = get_episode_links(episode_list)
    save_raw_scripts(episode_titles, episode_links, headers)
    df = create_transcript_file(episode_titles)
    df.to_csv("data/tbbt/tbbt_lines_v1.csv", index=False, encoding="utf-8")
    scene_count = df.groupby(["season", "episode"])["scene"].nunique().sort_values()
    logger.info(f"Scenes per episode - mean: {scene_count.mean()}")
    logger.info(f"Scenes per episode - median: {scene_count.median()}")

# pd.options.display.max_columns = 10
# pd.options.display.max_rows = None
#
# tbbt_df = pd.read_csv("../data/tbbt/tbbt_lines_v1.csv")
# scene_count = tbbt_df.groupby(["season", "episode"])["scene"].nunique().sort_values()
# print(scene_count)
# print(scene_count.mean())
# print(scene_count.median())

# with open(f"../data/tbbt/errors.txt", "r") as err:
#     x = len(err.readlines())
#     print('Not parsed lines:', x)
#
# for ep_title in episodes_titles:
#     with open(f"../data/tbbt/raw_scripts/{ep_title}.txt", "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             items = re.findall(r"(^\([^)]*\))$", line)
#             if items:
#                 with open("../data/tbbt/raw_scripts/stage_directions.txt", "a") as sd:
#                     sd.write(f"{ep_title} {items}\n")
#
# for ep_title in episodes_titles:
#     with open(f"../data/tbbt/raw_scripts/{ep_title}.txt", "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             items = re.findall(r"(\([^)]*\))", line)
#             if items:
#                 with open("../data/tbbt/raw_scripts/stage_directions2.txt", "a") as sd:
#                     sd.write(f"{ep_title} {items}\n")