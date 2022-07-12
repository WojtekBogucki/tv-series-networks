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

i = 195
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
        script_text = [p.get_text()
                           .replace("\n", " ")
                           .replace("\r", "")
                           .replace("<", "(")
                           .replace(">", ")")
                           .replace("\x97", " ")
                           .replace("\xa0", "")
                           .replace("\x91", "'")
                           .replace("\x92", "'")
                           .replace("\x93", "")
                           .replace("\ufffd", "")
                           .replace("\x85", "'").strip() + "\n" for p in
                       soup_subpage.body.findAll("p")]
    print(len(script_text))
    try:
        with open(f"../data/friends/raw_scripts/{ep_title}.txt", "w", encoding="utf-8") as f:
            f.writelines(script_text)
    except UnicodeEncodeError as uee:
        print(uee)
    if i >= 1: break

friends_df = pd.DataFrame(columns=["season", "episode", "title", "scene", "speaker", "line"])
# friends_df = pd.read_csv("../data/friends/friends_lines_v1.csv",encoding="utf-8")
seasons = []
episodes = []
titles = []
scenes = []
speakers = []
lines = []
scene = 0
for ep_title in episodes_titles:
    pattern = re.compile(r'^(\d{1,2})(\d{2})_(\d{3,4}_)?([&\w]+)', re.IGNORECASE)
    info = pattern.search(ep_title)
    season = int(info.group(1))
    episode = int(info.group(2))
    title = info.group(4)
    if (season == 2 and episode == 3) or (season == 2 and episode == 6) or (season == 9 and episode == 8):
        ep_title = "fixed/" + ep_title
    print(season, episode, title)
    with open(f"../data/friends/raw_scripts/{ep_title}.txt", "r", encoding="utf-8") as f:
        for line in f:
            full_line = line
            line = line.strip()
            stage_dirs = re.findall(r"(^[(\[][^)\]]*[)\]])$", line)
            if stage_dirs:
                for word in ["enters?", "exits?", "leaves?", "walks? in|out", "hungs up", "burts in", "approaches",
                             "comes? in"]:
                    match = re.findall(fr"(^[(\[].*{word}.*[)\]])$", line, re.IGNORECASE)
                    if match:
                        scene += 1
                        break
                if match:
                    continue
            # line = re.sub(r"([(\[][^)\]]*[)\]])", "", line)
            lower_line = line.lower()
            if "written by" in lower_line or \
                    lower_line.startswith("teleplay by:") or \
                    lower_line.startswith("story by:") or \
                    lower_line.startswith("transcriber's note") or \
                    lower_line.startswith("opening credits") or \
                    lower_line.startswith("opening titles") or \
                    lower_line.startswith("commercial break") or \
                    lower_line.startswith("closing credits") or \
                    lower_line.startswith("closing titles") or \
                    lower_line.startswith("the end") or \
                    lower_line.startswith("end"):
                continue
            elif lower_line.startswith("[scene") or \
                    lower_line.startswith("[time") or \
                    lower_line.startswith("[cut") or \
                    lower_line.startswith("[at") or \
                    lower_line.startswith("(at") or \
                    lower_line.startswith("[out") or \
                    lower_line.startswith("[back") or \
                    line.startswith("[later"):
                scene += 1
                continue
            line = re.sub(r"(\([^)]*\))", "", line)
            if not line:
                continue
            pattern = re.compile(r"(^[A-Za-z0-9'.#& \"’,]+): ? ?(.*)")
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
                continue
            if season == 9 and episode == 8:
                speaker = speaker.split(" ")[0]
            seasons.append(season)
            episodes.append(episode)
            titles.append(title)
            scenes.append(scene)
            speakers.append(speaker.strip().lower())
            lines.append(line.strip())

    # if episode >= 10: break
friends_df = pd.DataFrame.from_dict({"season": seasons,
                                     "episode": episodes,
                                     "title": titles,
                                     "scene": scenes,
                                     "speaker": speakers,
                                     "line": lines})
#
friends_df = friends_df[~((friends_df.season == 7) & (friends_df.episode == 24))]
friends_df.to_csv("../data/friends/friends_lines_v1.csv", index=False, encoding="utf-8")

pd.options.display.max_columns = 10
pd.options.display.max_rows = None
friends_df = pd.read_csv("../data/friends/friends_lines_v1.csv")
friends_df.groupby(["season", "episode"])["scene"].nunique().plot(kind="barh")
scene_count = friends_df.groupby(["season", "episode"])["scene"].nunique().sort_values()
print(scene_count)
print(scene_count.mean())
print(scene_count.median())

scene_count[:20]
scene_count[-20:]

friends_df.groupby(["season"])["scene"].nunique()
friends_df.groupby(["season", "episode"])["scene"].nunique()

friends_df2 = pd.read_csv("../data/friends/friends_r_package.csv")
friends_df2.groupby(["season", "episode"])["scene"].nunique().plot(kind="barh")
friends_df2.groupby(["season", "episode"])["scene"].nunique().sort_values()[:20]

######################
# from convokit import Corpus, download
#
# corpus = Corpus(filename=download("friends-corpus"))
# corpus.get_conversations_dataframe().to_csv("../data/friends/friends_corpus.csv", index=False, encoding="utf-8")
