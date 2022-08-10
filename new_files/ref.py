# from utilities import plot, fits, gmm, errors, predict, preprocess
import numpy as np
import time
from skgpytorch.metrics import mean_squared_error, negative_log_predictive_density
from gpytorch.constraints import GreaterThan
from skgpytorch.models import SVGPRegressor, SGPRegressor
from gpytorch.kernels import RBFKernel, ScaleKernel, PeriodicKernel
import torch
from datetime import datetime
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pandas as pd
from functools import partial
from sklearn.preprocessing import StandardScaler
import tqdm
from flax.training import train_state
import matplotlib.pyplot as plt
import seaborn as sns
# from utilities.fits import fit
# from utilities import plot
import tinygp
from tinygp import kernels, GaussianProcess
import tensorflow_probability.substrates.jax as tfp
import os
import argparse


parser = argparse.ArgumentParser()
# TODO: Add --model argument
# TODO: Add --n_inducing argument
parser.add_argument("--mode", choices=["train", "test"], default="test")
args = parser.parse_args()


class Config:
    repo_home = "."
    data_path = os.path.join(repo_home, "datasets")
    model_path = os.path.join(repo_home, "models_loaded")
    result_path = os.path.join(repo_home, "raw_results")


dist = tfp.distributions

# from datasets.dataset_load import dataset_loader

device = "cuda"

# XLA_PYTHON_CLIENT_PREALLOCATE=False


def load_data(args, iteration):
    # Load data
    if os.path.exists(os.path.join(Config.data_path, args.data)):
        X_train = pd.read_csv(
            os.path.join(Config.data_path, args.data,
                         "X_train_" + str(iteration) + ".csv.gz")
        )
        y_train = pd.read_csv(
            os.path.join(Config.data_path, args.data,
                         "y_train_" + str(iteration) + ".csv.gz")
        )
        X_train = torch.tensor(X_train.to_numpy()).float()
        y_train = torch.tensor(y_train.to_numpy()).float()
        if args.mode == "test":
            X_test = pd.read_csv(
                os.path.join(Config.data_path, args.data,
                             "X_test_" + str(iteration) + ".csv.gz")
            )
            y_test = pd.read_csv(
                os.path.join(Config.data_path, args.data,
                             "y_test_" + str(iteration) + ".csv.gz")
            )
            X_test = torch.tensor(X_test.to_numpy()).float()
            y_test = torch.tensor(y_test.to_numpy()).float()
            return (
                X_train.to(args.device),
                y_train.to(args.device).ravel(),
                X_test.to(args.device),
                y_test.to(args.device).ravel(),
            )
        return X_train.to(args.device), y_train.to(args.device).ravel()
    else:
        raise ValueError("Data not found at", Config.data_path)


kernel1 = ScaleKernel(RBFKernel(ard_num_dims=x_train.shape[1]))
kernel2 = ScaleKernel(RBFKernel(ard_num_dims=x_train.shape[1]))
kernel3 = ScaleKernel(PeriodicKernel(ard_num_dims=x_train.shape[1]))

kernel = kernel1 + kernel2*kernel3
induce_points = 512
inducing_points = x_train[torch.randperm(x_train.shape[0])[
    : induce_points]]
n_restarts = 1
if args.mode == "train":

    batch = 5000

    x_train = x_train[:45000, :]
    y_train = y_train[:45000]
    for i in range(2):

        model = SGPRegressor(x_train.to(device), y_train.to(device), kernel,
                             inducing_points).to(device)
        model.mll.likelihood.noise_covar.register_constraint(
            "raw_noise", GreaterThan(0.1))
        for param in model.parameters():
            torch.nn.init.normal_(param, 0, 1)
        start_time = time.time()
        model.fit(lr=0.1, n_epochs=100,
                  random_state=i, batch_size=None, n_restarts=n_restarts)
        end_time = time.time()

        torch.save(model.state_dict(), os.path.join(
            Config.model_path, f"SGPR_{induce_points}_{n_restarts}_{i}"))
        print("Training Time:", (end_time-start_time)/(60*n_restarts))

else:
    # i = 0
    y_pred_arr = []
    x_train = x_train[:45000, :]
    y_train = y_train[:45000]

    for i in range(1, 2):
        model = SGPRegressor(x_train.to(device), y_train.to(device), kernel,
                             inducing_points).to(device)
        model.load_state_dict(torch.load(
            os.path.join(Config.model_path, f"SGPR_{induce_points}_{n_restarts}_{i}")))

        with torch.no_grad():
            pred_dist = model.predict(x_test)

            y_pred_arr.append(pred_dist.loc)
        print(i)

    y_pred_arr1 = []
    for i in range(len(y_pred_arr)):
        y_pred_arr1.append(np.array(y_pred_arr[i].cpu()))

    y_mean = np.mean(y_pred_arr1, axis=0)
    y_sigma = np.std(y_pred_arr1, axis=0)

    y_mean.shape, y_sigma.shape

    y_mean = scaler_y.inverse_transform(y_mean.reshape(-1, 1)).squeeze()
    y_sigma = scaler_y.inverse_transform(y_sigma.reshape(-1, 1)).squeeze()
    print(y_test.shape, y_mean.shape)

    mae = np.mean(
        np.abs(np.asarray(y_test.cpu()).reshape(-1, 1) - y_mean.reshape(-1, 1)))
    # rms = errors.rmse(jnp.array(y_test.cpu()), y_mean)

    print("MAE: ", mae)
    # print("RMSE: ", rms)

    idx = 300
    plt.figure(figsize=(10, 6))
    plt.plot(jnp.arange(idx), y_test.cpu()[:idx], label="Refri")
    plt.plot(jnp.arange(idx), y_mean[:idx], label="Predicted")
    plt.legend(bbox_to_anchor=(1, 1), fontsize=20)
    sns.despine()
    plt.savefig("gp_ref_srpr_mean.png", bbox_inches="tight")

    idx = 300
    plt.figure(figsize=(10, 6))
    plt.plot(jnp.arange(idx), y_test.cpu()[
        :idx], label="Refrigerator", color="green")
    plt.plot(jnp.arange(idx), y_mean[:idx].reshape(-1, 1), label="Predicted")
    for i in range(1, 4):
        plt.fill_between(jnp.arange(idx), y_mean[:idx] - i*y_sigma[:idx], y_mean[:idx] + i*y_sigma[:idx],
                         color="orange", alpha=(1/(i*3)), label=f"$\mu\pm{i}*\sigma$")
    plt.legend(bbox_to_anchor=(1, 1), fontsize=20)
    plt.ylabel("Power", fontsize=20)
    sns.despine()
    plt.savefig("gp_ref_sgpr_both.png")
