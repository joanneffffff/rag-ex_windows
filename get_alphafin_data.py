import os
from config.parameters import Config

# cache_dir = 'D:/AI/huggingface/datasets'
cache_dir = Config().cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir

from datasets import load_dataset
import pathlib
import warnings

warnings.filterwarnings("ignore")

# Dataset name and expected cache folder
dataset_name = "AlphaFin/AlphaFin-dataset-v1"
cache_path = os.path.join(cache_dir, "AlphaFin___AlphaFin-dataset-v1")

# Load dataset
dataset = load_dataset(dataset_name, cache_dir=cache_dir)

# Show first sample
print(dataset)
print(dataset["train"][0])
