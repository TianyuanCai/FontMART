# FontMART

Personalized Font Recommendations: Combining ML and Typographic Guidelines to Optimize Readability

This program was developed using LightGBM on GPU. The code below applies more generally to CPU runtime.

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