import requests
import re
from bs4 import BeautifulSoup
import pandas as pd

main_URL = "https://bigbangtrans.wordpress.com/"
headers = {"User-Agent": "'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'"}
page = requests.get(main_URL, headers=headers)

soup = BeautifulSoup(page.content, "html.parser")
pages = soup.find_all("div", {"id": "pages-2"})[0]
elements = pages.findChildren("a")[1:]
episodes_link = list(map(lambda x: x.get("href").strip(), elements))
episodes_titles = list(
    map(lambda x: x.text
        .strip()
        .lower()
        .replace("\xa0", " ")
        .replace("series ", "s")
        .replace(" episode ", "e")
        .translate(str.maketrans({'\n': '', '–': '', ' ': '_', '(': '', ')': '', '/': ''}))
        .replace("__", "_"),
        elements))

i = 0
for ep_title, ep_link in zip(episodes_titles[i:], episodes_link[i:]):
    i += 1
    print(i)
    sub_page = requests.get(ep_link, headers=headers)
    soup_subpage = BeautifulSoup(sub_page.content, "html.parser")
    script_text = soup_subpage.find_all("div", {"class": "entrytext"})[0].get_text()
    script_text = re.sub(r'\n{2,}', '', script_text)
    script_text = re.sub(r'\nShare this:.*$', '', script_text)
    script_text = re.sub(r'\nWritten by.*$', '', script_text)
    try:
        with open(f"../data/tbbt/raw_scripts/{ep_title}.txt", "w", encoding="utf-8") as f:
            f.write(script_text)
    except UnicodeEncodeError as uee:
        print(uee)
    # if i >= 10: break

tbbt_df = pd.DataFrame(columns=["season", "episode", "title", "scene", "speaker", "line"])
# seinfeld_df = pd.read_csv("../data/seinfeld/seinfeld_lines_v1.csv",encoding="utf-8")
scene = 0
for ep_title in episodes_titles:
    pattern = re.compile(r'^s(\d{2})e(\d{2})_([a-z0-9_-]+)', re.IGNORECASE)
    info = pattern.search(ep_title)
    season = int(info.group(1))
    episode = int(info.group(2))
    title = info.group(3)
    print(season, episode)
    with open(f"../data/tbbt/raw_scripts/{ep_title}.txt", "r", encoding="utf-8") as f:
        for line in f:
            full_line = line
            line = line.strip()
            stage_dirs = re.findall(r"(\([^)]*\))", line)
            if stage_dirs:
                for word in ["enters?", "entering", "exits?", "exiting", "leaves?", "walks? in|out", "hungs up", "approaches"]:
                    match = re.findall(fr"(^[(\[].*{word}.*[)\]])$", line, re.IGNORECASE)
                    if match:
                        scene += 1
            line = re.sub(r"(\([^)]*\))", "", line)
            if not line:
                continue
            elif line.startswith("Scene:") or \
                    line.startswith("Fantasy sequence") or \
                    line.startswith("End fantasy") or \
                    line.startswith("Back to apartment") or \
                    line.startswith("Flash") or \
                    line.startswith("Time shift") or \
                    line.startswith("(Time shift") or \
                    line.startswith("(Later") or \
                    line.startswith("(Back") or \
                    line.startswith("Later"):
                scene += 1
                continue
            elif line.startswith("(") or line.startswith(".") or line.startswith("Credit"):
                continue
            pattern = re.compile(r"(^[A-Za-z0-9'.#& \"’]+): ? ?(.*)")
            line_search = pattern.search(line)
            if line_search is not None:
                speaker = line_search.group(1)
                line = line_search.group(2)
            else:
                with open(f"../data/tbbt/errors.txt", "a") as err:
                    err.write(f"{season} {episode} Line: {line}\n")
                # print("*error*", line)
                continue
            tbbt_df = tbbt_df.append({"season": season,
                                      "episode": episode,
                                      "title": title,
                                      "scene": scene,
                                      "speaker": speaker.strip().lower(),
                                      "line": line.strip()}, ignore_index=True)
    # if episode >= 1: break

tbbt_df.to_csv("../data/tbbt/tbbt_lines_v1.csv", index=False, encoding="utf-8")
pd.options.display.max_columns = 10
pd.options.display.max_rows = None

tbbt_df.groupby(["season", "episode"])["scene"].nunique().sort_values()

for ep_title in episodes_titles:
    with open(f"../data/tbbt/raw_scripts/{ep_title}.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            items = re.findall(r"(^\([^)]*\))$", line)
            if items:
                with open("../data/tbbt/raw_scripts/stage_directions.txt", "a") as sd:
                    sd.write(f"{ep_title} {items}\n")

for ep_title in episodes_titles:
    with open(f"../data/tbbt/raw_scripts/{ep_title}.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            items = re.findall(r"(\([^)]*\))", line)
            if items:
                with open("../data/tbbt/raw_scripts/stage_directions2.txt", "a") as sd:
                    sd.write(f"{ep_title} {items}\n")