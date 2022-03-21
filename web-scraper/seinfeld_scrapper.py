'''
Based on: https://github.com/luonglearnstocode/Seinfeld-text-corpus/blob/master/scraper.ipynb
'''
import requests
import re
from bs4 import BeautifulSoup
import pandas as pd

######## OLD VERSION
# main_URL = "https://www.seinfeldscripts.com/"
# URL = main_URL + "seinfeld-scripts.html"
# headers = {"User-Agent": "'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0'"}
# page = requests.get(URL, headers=headers)
#
# soup = BeautifulSoup(page.content, "html.parser")
#
# tables = soup.find_all("table")
#
# episodes_list = tables[1].findChildren('a')
# episodes_link = list(map(lambda x: main_URL + x.get('href').strip(), episodes_list))
# episodes_title = list(
#     map(lambda x: x.text.strip().lower().translate(str.maketrans({'\n': '', ' ': '_', '(': '', ')': ''})),
#         episodes_list))
#
# i = 0
# for ep_title, ep_link in zip(episodes_title[i:], episodes_link[i:]):
#     i += 1
#     print(i)
#     sub_page = requests.get(ep_link, headers=headers)
#     soup_subpage = BeautifulSoup(sub_page.content, "html.parser")
#     soup_subpage = soup_subpage(id="content")[0]
#     soup_subpage_p = soup_subpage.find_all("p", recursive=False)
#     soup_subpage_p = [x for x in soup_subpage_p if not len(x.findChildren())]
#     script_text = '\n'.join(list(map(lambda x: x.text, soup_subpage_p)))
#     script_text = re.sub('\\n{2,}', '', script_text)
#     try:
#         with open("data/seinfeld/{0}{1}.txt".format(i, ep_title), "w", encoding="utf-8") as f:
#             f.write(script_text)
#     except UnicodeEncodeError as uee:
#         print(uee)
#     if i >= 10: break

######################
main_URL = "https://web.archive.org"
URL = "https://web.archive.org/web/20170504050113/http://www.seinology.com/scripts-english.shtml"
headers = {"User-Agent": "'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0'"}
page = requests.get(URL, headers=headers)

soup = BeautifulSoup(page.content, "html.parser")
tables = soup.find_all("table")

episodes_list = tables[2].findChildren('a')
# get episode titles
episodes_titles = [x.text for x in episodes_list if
                   str(x.get("href")).startswith("/web") or str(x.get("href")).startswith("scripts") and str(
                       x.get("href")).endswith(".shtml")]
episodes_titles = list(
    map(lambda x: x.translate(str.maketrans({'\n': '', ' ': '_', '(': '', ')': '', ',': ''})),
        episodes_titles))
# get episode links
episodes_links1 = [x.get("href") for x in episodes_list if str(x.get("href")).startswith("/web")]
episodes_links2 = ['/web/20170504050113/http://www.seinology.com/' + x.get("href") for x in episodes_list if
                   str(x.get("href")).startswith("scripts") and str(x.get("href")).endswith(".shtml")]
episodes_links = episodes_links1 + episodes_links2
episodes_links = list(map(lambda x: main_URL + x, episodes_links))

i = 0
for ep_title, ep_link in zip(episodes_titles[i:], episodes_links[i:]):
    i += 1
    print(i)
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
        print("Episode not saved")
        continue
    script_text = re.sub(r'\t*', '', script_text)
    script_text = re.sub(r'[\r\n]{2,}', '\n', script_text)
    # script_text = re.sub(r'\s{2,}', ' ', script_text)
    script_text = re.sub(r'(“|”)', r'"', script_text)
    script_text = re.sub(r'(‘|’)', r"'", script_text)
    if i == 1:
        script_text = re.sub('KESSLER', 'KRAMER', script_text)
        script_text = re.sub('Kessler', 'Kramer', script_text)
    try:
        with open(f"../data/seinfeld/seinology/{ep_title}.txt", "w", encoding="utf-8") as f:
            f.write(script_text.strip())
    except UnicodeEncodeError as uee:
        print(uee)
    except ConnectionError as te:
        print(te)
    # if i >= 1: break

seinfeld_df = pd.DataFrame(columns=["season", "episode", "title", "scene", "speaker", "line"])
# seinfeld_df = pd.read_csv("../data/seinfeld/seinfeld_lines_v1.csv",encoding="utf-8")
season = 1
scene = 0
episode = 0
for ep_title in episodes_titles:
    pattern = re.compile(r'^(\d+)-(\w+)', re.IGNORECASE)
    ep_number = int(pattern.search(ep_title).group(1))
    title = pattern.search(ep_title).group(2)
    print(ep_number)
    if ep_number in [6, 18, 41, 65, 87, 111, 135, 157]:
        season += 1
        episode = 0
    elif ep_number in [47, 116, 121]:
        ep_title = "fixed/" + ep_title
    elif ep_number in [100, 177]:  # recap episodes
        episode += 1
        continue
    elif ep_number in [180, 101, 178]:  # second parts
        continue
    with open(f"../data/seinfeld/seinology/{ep_title}.txt", "r", encoding="utf-8") as f:
        if not title.endswith("2"):
            episode += 1
        if title.endswith("1") or title.endswith("2"):
            title = title[:-1]
        for line in f:
            full_line = line
            line = line.strip()
            stage_dirs = re.findall(r"(^\([^)]*\))$", line)
            if stage_dirs:
                for word in ["enters?", "exits?", "leaves?", "walks? in|out", "hungs up", "bursts in", "approaches",
                             "comes? in"]:
                    match = re.findall(fr"(^\(.*{word}.*\))$", line, re.IGNORECASE)
                    if match:
                        scene += 1
                        continue
            line = re.sub(r"(\([^)]*\))", "", line)
            for name in ["GEORGE", "JERRY", "KRAMER", "ELAINE"]:
                line = re.sub(fr"(?<=^{name})  ", ": ", line)
            if not line:
                continue
            elif line.startswith("*") or line.startswith("Notice") or line.startswith("(") or line.startswith("%"):
                continue
            elif line.startswith("INT.") or line.startswith("EXT.") or line.startswith("["):
                scene += 1
                continue
            pattern = re.compile(r"(^[A-Za-z0-9'.#& \"-]{,30}): ? ?(.*)")
            line_search = pattern.search(line)
            if line_search is not None:
                speaker = line_search.group(1)
                line = line_search.group(2)
            else:
                with open(f"../data/seinfeld/seinology/errors.txt", "a") as err:
                    err.write(f"{ep_number} Line: {line}\n")
                # print("*error*", line)
                continue
            if not speaker.strip():
                continue
            seinfeld_df = seinfeld_df.append({"season": season,
                                              "episode": episode,
                                              "title": title,
                                              "scene": scene,
                                              "speaker": speaker.strip().lower(),
                                              "line": line.strip()}, ignore_index=True)
    # if ep_number >= 1: break

# removing recap or duplicated episodes
seinfeld_df.to_csv("../data/seinfeld/seinfeld_lines_v1.csv", index=False, encoding="utf-8")

seinfeld_df.groupby(["season", "episode"])["scene"].nunique().plot(kind="barh")
seinfeld_df.groupby(["season", "episode"])["scene"].nunique().sort_values()

for ep_title in episodes_titles:
    with open(f"../data/seinfeld/seinology/{ep_title}.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            items = re.findall(r"(^\([^)]*\))$", line)
            if items:
                with open("../data/seinfeld/seinology/stage_directions.txt", "a") as sd:
                    sd.write(f"{ep_title} {items}\n")

i = 0
for ep_title in episodes_titles:
    with open(f"../data/seinfeld/seinology/{ep_title}.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            items = re.findall(r"(^\(.*enters?.*\))$", line)
            if items:
                i += 1
                print(f"{i} {items}")

# enters, enter, walks in, walk in, walks out, leaves, leave, hungs up, burts in, approaches, exit
