from __future__ import annotations

import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


class Font:
    def __init__(self, font_file, font_size=12):
        self.font_name = font_file.split('/')[-1].split('.')[0]
        self.font = ImageFont.truetype(font_file, font_size)

    def pixel_wh(self, unicode_text):
        width = len(unicode_text) * 50
        height = 100
        back_ground_color = (0, 0, 0)
        font_color = (255, 255, 255)

        im = Image.new('RGB', (width, height), back_ground_color)
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), unicode_text, font=self.font, fill=font_color)

        tmp_img_file = 'tmp_text.png'
        im.save(tmp_img_file)
        box = Image.open(tmp_img_file).getbbox()
        return box[2] - box[0], box[3] - box[1]

    def grayscale(self, unicode_text):
        # count greyscale
        width = len(unicode_text) * 50
        height = 100
        back_ground_color = (0, 0, 0)
        font_color = (255, 255, 255)

        im = Image.new('RGB', (width, height), back_ground_color)
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), unicode_text, font=self.font, fill=font_color)

        tmp_img_file = 'tmp_text.png'
        im.save(tmp_img_file)
        box = Image.open(tmp_img_file).getbbox()

        # crop to get box size
        im = im.convert('L')

        return np.count_nonzero(im) / ((box[2] - box[0]) * (box[3] - box[1]))

    def font_metrics(self):
        text = string.ascii_lowercase
        text_upper = string.ascii_uppercase
        text_base = [c for c in text if c not in 'bdfhijkltgqpy']

        # width
        avg_char_width_lower = np.sum(
            [self.pixel_wh(l)[0] for l in text],
        ) / len(text)
        avg_char_width_lower_sd = np.std([self.pixel_wh(l)[0] for l in text])

        # char horizontal spacing
        avg_char_spacing_lower = (
            self.pixel_wh(text)[0] - avg_char_width_lower * len(text)
        ) / (len(text) - 1)

        # height
        avg_char_height_lower = np.sum([self.pixel_wh(l)[1] for l in text_base]) / len(
            text_base,
        )

        avg_char_height_upper = np.sum(
            [self.pixel_wh(l)[1] for l in text_upper],
        ) / len(text_upper)
        ascent, descent = self.font.getmetrics()
        grayscale = self.grayscale(text)

        print(
            self.font_name,
            avg_char_spacing_lower,
            avg_char_width_lower,
            avg_char_spacing_lower / avg_char_width_lower,
        )

        return {
            'avg_char_width_lower': avg_char_width_lower,
            'avg_char_width_lower_sd': avg_char_width_lower_sd,
            'avg_char_spacing_lower': avg_char_spacing_lower,
            'avg_char_spacing_lower_relative': avg_char_spacing_lower
            / avg_char_width_lower,
            'avg_char_height_lower': avg_char_height_lower,
            'avg_char_height_upper': avg_char_height_upper,
            'ascent': ascent,
            'descent': descent,
            'grayscale': grayscale,
        }


def save_old_font_metrics():
    font_dir = 'data/raw/fonts/fonts_ttf'
    font_file_list = []
    for path, subdirs, files in os.walk(font_dir):
        for name in files:
            if 'ttf' in name:
                font_file_list.append(os.path.join(path, name))

    # calculate font metrics, but only using size-invariant measure
    font_list, font_metrics_dict = [], []
    for f in font_file_list:
        font_obj = Font(font_file=f)
        font_metrics_dict.append(font_obj.font_metrics())
        font_list.append(f.split('.')[0])

    df = pd.DataFrame(font_metrics_dict)
    df['font'] = [x.split('/')[-1] for x in font_list]

    js_metrics = pd.read_csv('data/processed/font_metrics_js.csv')
    df = (
        df[['font', 'grayscale']]
        .merge(js_metrics, left_on='font', right_on='font-name')
        .drop('font-name', axis=1)
    )

    df.to_csv('data/processed/font_metrics_js_combo_old.csv', index=False)


if __name__ == '__main__':
    font_dir = 'data/raw/new_fonts'
    font_file_list = os.listdir(font_dir)

    # calculate font metrics, but only using size-invariant measure
    font_list, font_metrics_dict = [], []
    for f in font_file_list:
        font_obj = Font(font_file=f'{font_dir}/{f}')
        font_metrics_dict.append(font_obj.font_metrics())
        font_list.append(f.split('.')[0])

    df = pd.DataFrame(font_metrics_dict)
    df['font'] = font_list

    js_metrics = pd.read_csv('data/processed/font_metrics_js.csv')
    df = (
        df[['font', 'grayscale']]
        .merge(js_metrics, left_on='font', right_on='font-name')
        .drop('font-name', axis=1)
    )

    df.to_csv('data/processed/font_metrics_js_combo.csv', index=False)

    # similarity analysis
    df = df.set_index(df['font']).drop('font', axis=1)

    plt.close('all')
    df.hist(bins=8)
    plt.tight_layout()
    plt.savefig('reports/input/font_features.png')

    sns.clustermap(df, standard_scale=1)
    plt.savefig('reports/input/font_cluster.png')
