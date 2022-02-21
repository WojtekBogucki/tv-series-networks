import requests
import re
from bs4 import BeautifulSoup
import pandas as pd

main_URL = "https://fangj.github.io/friends/"
headers = {
    "User-Agent": "'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'"}
page = requests.get(main_URL, headers=headers)

soup = BeautifulSoup(page.content, "html.parser")
elements = soup.findChildren("a")
episodes_link = list(map(lambda x: main_URL + x.get("href").strip(), elements))
episodes_titles = list(
    map(lambda x: x.text
        .strip()
        .lower()
        .translate(
        str.maketrans({':': '', '–': '', '-': '_', ' ': '_', '(': '', ')': '', ',': '', "'": '', '"': '', '.': ''}))
        .replace("__", "_"),
        elements))

i = 0
for ep_title, ep_link in zip(episodes_titles[i:], episodes_link[i:]):
    i += 1
    print(i, ep_title)
    sub_page = requests.get(ep_link, headers=headers)
    soup_subpage = BeautifulSoup(sub_page.text, "lxml")
    if i in [28, 30, 31, 33] + [*range(36, 48)]:
        script_text = [p.replace("\n", " ").replace("\x97", " ").replace("\xa0", "").strip() + "\n" for p in
                       soup_subpage.body.get_text().split("\n\n") if p.strip()]
    elif i in [199, 203]:
        script_text = []
        for p in str(soup_subpage.body).split("<br/>"):
            p = p.replace("\n", " ").replace("\x97", " ").replace("\xa0", "").replace("\x85", "")
            p = re.sub(r"</?\w+/?>", "", p)
            p = p.strip()
            if p and not p.startswith("<"):
                script_text.append(p + "\n")
    else:
        script_text = [p.get_text().replace("\n", " ").replace("\x97", " ").replace("\xa0", "").replace("\x91", "'").replace("\x92", "'").replace("\x85", "'").strip() + "\n" for p in
                       soup_subpage.body.findAll("p")]
    print(len(script_text))
    try:
        with open(f"../data/friends/raw_scripts/{ep_title}.txt", "w", encoding="utf-8") as f:
            f.writelines(script_text)
    except UnicodeEncodeError as uee:
        print(uee)
    # if i >= 1: break

friends_df = pd.DataFrame(columns=["season", "episode", "title", "scene", "speaker", "line"])
# friends_df = pd.read_csv("../data/friends/friends_lines_v1.csv",encoding="utf-8")
scene = 0
for ep_title in episodes_titles:
    pattern = re.compile(r'^(\d{1,2})(\d{2})_(\d{3,4}_)?([&\w]+)', re.IGNORECASE)
    info = pattern.search(ep_title)
    season = int(info.group(1))
    episode = int(info.group(2))
    title = info.group(4)
    print(season, episode, title)
    with open(f"../data/friends/raw_scripts/{ep_title}.txt", "r", encoding="utf-8") as f:
        for line in f:
            full_line = line
            line = line.strip()
            line = re.sub(r"(\([^)]*\))", "", line)
            lower_line = line.lower()
            if not line:
                continue
            elif lower_line.startswith("written by:") or \
                    lower_line.startswith("opening credits") or \
                    lower_line.startswith("opening titles") or \
                    lower_line.startswith("commercial break") or \
                    lower_line.startswith("closing credits") or \
                    lower_line.startswith("end"):
                continue
            elif lower_line.startswith("[scene") or \
                    lower_line.startswith("[time") or \
                    lower_line.startswith("[cut") or \
                    line.startswith("[later"):
                scene += 1
                continue
            pattern = re.compile(r"([A-Za-z0-9'.#& \"’]+): ? ?(.*)")
            line_search = pattern.search(line)
            if line_search is not None:
                speaker = line_search.group(1)
                line = line_search.group(2)
            else:
                try:
                    with open(f"../data/friends/errors.txt", "a") as err:
                        err.write(f"{season} {episode} Line: {line}\n")
                except UnicodeEncodeError as uee:
                    print(uee)
                # print("*error*", line)
                continue
            friends_df = friends_df.append({"season": season,
                                      "episode": episode,
                                      "title": title,
                                      "scene": scene,
                                      "speaker": speaker.strip().lower(),
                                      "line": line.strip()}, ignore_index=True)
    # if episode >= 10: break
#
# seinfeld_df = seinfeld_df[~seinfeld_df.episode.isin([180, 101, 100, 177, 178])]
friends_df.to_csv("../data/friends/friends_lines_v1.csv", index=False, encoding="utf-8")
