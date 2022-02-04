import requests
import re
from bs4 import BeautifulSoup
import pandas as pd

main_URL = "https://bigbangtrans.wordpress.com/"
headers = {"User-Agent": "'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0'"}
page = requests.get(main_URL, headers=headers)

soup = BeautifulSoup(page.content, "html.parser")
pages = soup.find_all("div", {"id": "pages-2"})[0]
elements = pages.findChildren("a")[1:]
episodes_link = list(map(lambda x: x.get("href").strip(), elements))
episodes_title = list(
    map(lambda x: x.text
        .strip()
        .lower()
        .replace("\xa0", " ")
        .replace("series ", "s")
        .replace(" episode ", "e")
        .translate(str.maketrans({'\n': '', '–': '', ' ': '_', '(': '', ')': '', '/': ''}))
        .replace("__", "_"),
        elements))

i = 124
for ep_title, ep_link in zip(episodes_title[i:], episodes_link[i:]):
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
