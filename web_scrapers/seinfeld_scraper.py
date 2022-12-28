'''
Based on: https://github.com/luonglearnstocode/Seinfeld-text-corpus/blob/master/scraper.ipynb
'''
import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

main_URL = "https://web.archive.org"
URL = "https://web.archive.org/web/20170504050113/http://www.seinology.com/scripts-english.shtml"
headers = {"User-Agent": "'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0'"}


def get_episode_list(url: str, headers: dict) -> list:
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")
    tables = soup.find_all("table")
    episode_list = tables[2].findChildren('a')
    logger.info("Got episode list")
    return episode_list


def get_episode_titles(episode_list: list) -> list:
    episodes_titles = [x.text for x in episode_list if
                       str(x.get("href")).startswith("/web") or str(x.get("href")).startswith("scripts") and str(
                           x.get("href")).endswith(".shtml")]
    episodes_titles = list(
        map(lambda x: x.translate(str.maketrans({'\n': '', ' ': '_', '(': '', ')': '', ',': ''})),
            episodes_titles))
    logger.info(f"Got {len(episodes_titles)} episode titles")
    return episodes_titles


def get_episode_links(episode_list: list, main_url: str) -> list:
    episodes_links1 = [x.get("href") for x in episode_list if str(x.get("href")).startswith("/web")]
    episodes_links2 = ['/web/20170504050113/http://www.seinology.com/' + x.get("href") for x in episode_list if
                       str(x.get("href")).startswith("scripts") and str(x.get("href")).endswith(".shtml")]
    episodes_links = episodes_links1 + episodes_links2
    episodes_links = list(map(lambda x: main_url + x, episodes_links))
    logger.info(f"Got {len(episodes_links)} episode links")
    return episodes_links


def save_raw_scripts(episode_titles: list, episode_links: list, headers: dict, i: int = 0) -> None:
    for ep_title, ep_link in zip(episode_titles[i:], episode_links[i:]):
        i += 1
        logger.info(f"Episode {i}, {ep_title}")
        sub_page = requests.get(ep_link, headers=headers, allow_redirects=True)
        soup_subpage = BeautifulSoup(sub_page.content, "html.parser")
        script_text = soup_subpage.text
        if i in [17, 41, 45, 63, 133, 131, 164]:
            script_text = re.sub(r'END OF SHOW|\(To be continued.*\)|To be continued.*', 'The End', script_text,
                                 flags=re.IGNORECASE)
        if i == 64:
            script_text = re.sub(r'(===\[)|(\]={3,})', '', script_text)
        if i == 72:
            script_text = re.sub(r'\tTHE END', '', script_text)
            script_text = re.sub(r'TRUELY THE END', 'THE END', script_text)
        if i in [82, 83]:
            script_text = re.sub(r'(Copyright 2006 seinology)', r'The End\n\1', script_text)
        if i in [115, 144]:
            script_text = script_text.replace('\xa0', '')
        if i == 119:
            script_text = re.sub(r'\(IMG.*\)', '', script_text)
        if i == 123:
            script_text = re.sub(r'<IMG.*>', '', script_text)
        pattern = re.compile(r'={30,}\n([^=]*)the end\W*\n', re.IGNORECASE)
        match = pattern.search(script_text)
        if match is not None:
            script_text = match.group(1)
        else:
            logger.warning(f"Episode {i} not saved")
            continue
        script_text = re.sub(r'\t*', '', script_text)
        script_text = re.sub(r'[\r\n]{2,}', '\n', script_text)
        script_text = re.sub(r'(“|”)', r'"', script_text)
        script_text = re.sub(r'(‘|’)', r"'", script_text)
        if i == 1:
            script_text = re.sub('KESSLER', 'KRAMER', script_text)
            script_text = re.sub('Kessler', 'Kramer', script_text)
        try:
            with open(f"data/seinfeld/seinology/{ep_title}.txt", "w", encoding="utf-8") as f:
                f.write(script_text.strip())
        except UnicodeEncodeError as uee:
            logger.error(uee)
        except ConnectionError as te:
            logger.error(te)
    logger.info("Finished saving raw scripts")


def create_transcript_file(episode_titles: list) -> pd.DataFrame:
    entry_words = ["enters?", "walks? in", "picks up", "bursts in", "approaches", "comes? in", "comes? over",
                   "shows? up"]
    exit_words = ["exits?", "leaves?", "walks? out", "hungs up"]
    new_scene_words = entry_words + exit_words
    next_scene = False
    seasons = []
    episodes = []
    titles = []
    scenes = []
    speakers = []
    lines = []
    season = 1
    scene = 0
    episode = 0
    errors_path = "data/seinfeld/seinology/errors.txt"
    if os.path.exists(errors_path):
        os.remove(errors_path)
    for ep_title in episode_titles:
        pattern = re.compile(r'^(\d+)-(\w+)', re.IGNORECASE)
        ep_number = int(pattern.search(ep_title).group(1))
        title = pattern.search(ep_title).group(2)
        logger.info(f"Episode: {ep_number}, {title}")
        if ep_number in [6, 18, 41, 65, 87, 111, 135, 157]:
            season += 1
            episode = 0
        elif ep_number in [16, 47, 54, 116, 121]:
            ep_title = "fixed/" + ep_title
        elif ep_number in [100, 177]:  # recap episodes
            episode += 1
            continue
        elif ep_number in [83, 180, 101, 178]:  # second parts
            continue
        with open(f"data/seinfeld/seinology/{ep_title}.txt", "r", encoding="utf-8") as f:
            if not title.endswith("2"):
                episode += 1
            if title.endswith("1") or title.endswith("2"):
                title = title[:-1]
            for line in f:
                full_line = line
                line = line.strip()
                if next_scene:
                    scene += 1
                    next_scene = False
                # find stage directions
                stage_dirs = re.findall(r"(\([^)]+\))", line)
                if stage_dirs:
                    for word in new_scene_words:
                        match = re.findall(fr"(\(.*{word}.*\))", line, re.IGNORECASE)
                        if match:
                            if word in entry_words or (word in exit_words and line.startswith("(")):
                                scene += 1
                                break
                            elif word in exit_words and not line.startswith("("):
                                next_scene = True
                # remove stage directions
                line = re.sub(r"(\([^)]+\))", "", line)
                for name in ["GEORGE", "JERRY", "KRAMER", "ELAINE"]:
                    line = re.sub(fr"(?<=^{name})  ", ": ", line)
                if not line:
                    continue
                elif line.startswith("*") or line.startswith("Notice") or line.startswith("%"):
                    continue
                elif line.startswith("INT.") or line.startswith("EXT.") or line.startswith("["):
                    scene += 1
                    continue
                pattern = re.compile(r"(^[A-Za-z0-9'.,#& \"-]{,30}): ? ?(.*)")
                line_search = pattern.search(line)
                if line_search is not None:
                    speaker = line_search.group(1)
                    line = line_search.group(2)
                else:
                    with open(errors_path, "a") as err:
                        err.write(f"{ep_number} Line: {line}\n")
                    # print("*error*", line)
                    continue
                if not speaker.strip():
                    continue
                seasons.append(season)
                episodes.append(episode)
                titles.append(title)
                scenes.append(scene)
                speakers.append(speaker.strip().lower())
                lines.append(line.strip())
    seinfeld_df = pd.DataFrame.from_dict({"season": seasons,
                                          "episode": episodes,
                                          "title": titles,
                                          "scene": scenes,
                                          "speaker": speakers,
                                          "line": lines})
    logger.info("Finished creating transcript file")
    return seinfeld_df


def run_seinfeld_scrapper():
    episode_list = get_episode_list(URL, headers)
    episode_titles = get_episode_titles(episode_list)
    episode_links = get_episode_links(episode_list, main_URL)
    save_raw_scripts(episode_titles, episode_links, headers)
    df = create_transcript_file(episode_titles)
    df.to_csv("data/seinfeld/seinfeld_lines_v1.csv", index=False, encoding="utf-8")
    scene_count = df.groupby(["season", "episode"])["scene"].nunique().sort_values()
    logger.info(f"Scenes per episode - mean: {scene_count.mean()}")
    logger.info(f"Scenes per episode - median: {scene_count.median()}")


if __name__ == "__main__":
    run_seinfeld_scrapper()
# seinfeld_df = pd.read_csv("../data/seinfeld/seinfeld_lines_v1.csv")
# seinfeld_df.groupby(["season", "episode"])["scene"].nunique().plot(kind="barh")

#
# for ep_title in episodes_titles:
#     print(ep_title)
#     with open(f"../data/seinfeld/seinology/{ep_title}.txt", "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             items = re.findall(r"(\([^)]*\))", line)
#             if items:
#                 with open("../data/seinfeld/seinology/stage_directions.txt", "a") as sd:
#                     sd.write(f"{ep_title} {items}\n")
#
# i = 0
# for ep_title in episodes_titles:
#     with open(f"../data/seinfeld/seinology/{ep_title}.txt", "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             items = re.findall(r"(^\(.*enters?.*\))$", line)
#             if items:
#                 i += 1
#                 print(f"{i} {items}")

# enters, enter, walks in, walk in, walks out, leaves, leave, hungs up, burts in, approaches, exit
