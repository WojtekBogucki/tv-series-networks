# NOT USED because dataset is on kaggle
# import requests
# import numpy as np
# from bs4 import BeautifulSoup
#
# main_URL = "https://transcripts.foreverdreaming.org/"
# start = np.arange(0,176,25)
# URL = main_URL + "viewforum.php?f=574"
# headers = {"User-Agent": "'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0'"}
# page = requests.get(URL, headers=headers)
#
# soup = BeautifulSoup(page.content, "html.parser")
#
# topictitle = soup.find_all('a', {'class':'topictitle'})[1:]
#
# episodes_link = list(map(lambda x: main_URL + x.get('href').strip()[2:], topictitle))
# episodes_title = list(
#     map(lambda x: x.text.strip().lower().translate(str.maketrans({'\n': '','-': '', ' ': '_', '(': '', ')': ''})),
#         topictitle))