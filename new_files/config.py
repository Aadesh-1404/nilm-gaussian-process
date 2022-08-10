import os
import torch


class Config:
    repo_home = "."
    data_path = os.path.join(repo_home, "datasets")
    model_path = os.path.join(repo_home, "models")
    result_path = os.path.join(repo_home, "raw_results")
