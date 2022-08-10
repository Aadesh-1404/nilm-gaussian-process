import os
from config import Config
import pandas as pd
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel, PeriodicKernel
from skgpytorch.models import ExactGPRegressor, SGPRegressor, SVGPRegressor
import numpy as np
from memory_profiler import profile


# @profile
def load_data(args):
    # Load data
    if os.path.exists(os.path.join(Config.data_path, args.appliance)):
        X_train = pd.read_csv(
            os.path.join(Config.data_path, args.appliance,
                         "x_train.csv.gz")
        )
        y_train = pd.read_csv(
            os.path.join(Config.data_path, args.appliance,
                         "y_train.csv.gz")
        )
        X_train = torch.tensor(X_train.to_numpy()).float()
        y_train = torch.tensor(y_train.to_numpy()).float()
        if args.mode == "test":
            X_test = pd.read_csv(
                os.path.join(Config.data_path, args.appliance,
                             "x_test.csv.gz")
            )
            y_test = pd.read_csv(
                os.path.join(Config.data_path, args.appliance,
                             "y_test.csv.gz")
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


# TODO: Include dataset size
def get_model_name(args, iter):
    model_name = (
        "train_{appliance}_{model}_{batch_size}_{device}_{lr}_{epochs}_{seed}_{extra}_{n_cpus}_{iter}.pt".format(  # _{appliance_size}
            appliance=args.appliance,
            model=args.model,
            batch_size=args.batch_size,
            device=args.device,
            lr=args.lr,
            epochs=args.epochs,
            seed=args.seed,
            extra=args.extra,
            n_cpus=args.n_cpus,
            iter=iter
        )
    )
    return model_name


def get_kernel(x_train):

    kernel1 = ScaleKernel(RBFKernel(ard_num_dims=x_train.shape[1]))
    kernel2 = ScaleKernel(RBFKernel(ard_num_dims=x_train.shape[1]))
    kernel3 = ScaleKernel(PeriodicKernel(ard_num_dims=x_train.shape[1]))

    kernel = kernel1 + kernel2*kernel3
    return kernel


def get_model(X_train, y_train, args):
    kernel = get_kernel(X_train)
    if args.model == "EGPR" or args.model == "SGDGP":
        model = ExactGPRegressor(X_train, y_train, kernel).to(args.device)
    elif args.model == "SGPR":
        # inducing_points = X_train[np.arange(0, X_train.shape[0], 20)]
        inducing_points = X_train[torch.randperm(
            X_train.shape[0])[: args.n_inducing]]
        model = SGPRegressor(X_train, y_train, kernel,
                             inducing_points).to(args.device)
    elif args.model == "SVGP":
        # inducing_points = X_train[np.arange(0, X_train.shape[0], 20)]
        inducing_points = X_train[torch.randperm(
            X_train.shape[0])[: args.n_inducing]]
        model = SVGPRegressor(X_train, y_train, kernel,
                              inducing_points).to(args.device)
    else:
        raise ValueError("Model not found")

    return model
