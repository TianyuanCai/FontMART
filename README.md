# FontMART

## Usage

### Docker

```
docker build -t fontmart . && docker run -v $(PWD):/app fontmart
```

You can also directly pull the [docker image](https://hub.docker.com/repository/docker/tianyuancai/fontmart) from Docker Hub.

### Conda

Create the conda environment with the `environment.yml`. Then run `python main.py`.

Conda is only needed to simply LightGBM installation. Therefore, you can alternatively take the following steps:

```
conda create -n fontmart PYTHON==3.9
conda activate fontmart
conda install lightgbm -y
pip install -r requirements.txt
python main.py
```

## Citation
```
@inproceedings{cai2022personalized,
  author = {Cai, Tianyuan and Wallace, Shaun and Rezvanian, Tina and Dobres, Jonathan and Kerr, Bernard and Berlow, Samuel and Huang, Jeff and Sawyer, Ben D. and Bylinskii, Zoya},
  title = {Personalized Font Recommendations: Combining ML and Typographic Guidelines to Optimize Readability},
  year = {2022},
  isbn = {9781450393584},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3532106.3533457},
  doi = {10.1145/3532106.3533457},
  abstract = { The amount of text people need to read and understand grows daily. Software defaults, designers, or publishers often choose the fonts people read in. However, matching individuals with a faster font could help them cope with information overload. We collaborated with typographers to (1) select eight fonts designed for digital reading to systematically compare their effectiveness and to (2) understand how font and reader characteristics affect reading speed. We collected font preferences, reading speeds, and characteristics from 252 crowdsourced participants in a remote readability study. We use font and reader characteristics to train FontMART, a learning to rank model that automatically orders a set of eight fonts per participant by predicted reading speed. FontMART’s fastest font prediction shows an average increase of 14–25 WPM compared to other font defaults, without hindering comprehension. This encouraging evidence provides motivation for adding our personalized font recommendation to future interactive systems.},
  booktitle = {Designing Interactive Systems Conference},
  pages = {1–25},
  numpages = {25},
  keywords = {readability, reading, typography, personalization},
  location = {Virtual Event, Australia},
  series = {DIS ‘22}
}
```
