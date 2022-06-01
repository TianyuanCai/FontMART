from __future__ import annotations

import os

import pandas as pd
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

df = pd.read_csv('data/raw/Speed Passages Study 4 - Sheet1.csv')

options = Options()
options.add_argument('--headless')  # Runs Chrome in headless mode.
options.add_argument('--no-sandbox')  # Bypass OS security model
options.add_argument('--disable-gpu')  # applicable to windows os only
options.add_argument('start-maximized')  #
options.add_argument('disable-infobars')
options.add_argument('--disable-extensions')
options.add_argument('--window-size=1920,1080')
driver = webdriver.Chrome(
    ChromeDriverManager().install(),
    chrome_options=options,
)

# get unique links and font pairs
df.columns = ['font', 'id_passage', 'iteration', 'url']

file_names = []
for i in tqdm(range(len(df))):
    # matchup a
    font = df.loc[i, 'font']
    url = df.loc[i, 'url']
    font_path = f'data/raw/paragraphs_new/{font}'
    os.makedirs(font_path, exist_ok=True)

    # get image
    driver.get(url)
    file_name = f"{font_path}/{df.loc[i, 'id_passage']}_{df.loc[i, 'iteration']}.jpg"
    file_names.append(file_name)
    driver.get_screenshot_as_file(file_name)

    # crop
    original = Image.open(file_name)
    original = original.convert('RGB')
    original = original.crop((35, 141, 956, 956))
    original.save(file_name)
