import os
from config import Config
import torch
import argparse
import pandas as pd
from helper import load_data, get_model_name, get_model
from gpytorch.constraints import GreaterThan
from skgpytorch.metrics import mean_squared_error, negative_log_predictive_density
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
# TODO: Add --model argument
# TODO: Add --n_inducing argument
parser.add_argument("--mode", choices=["train", "test"], default="test")
parser.add_argument("--batch_predict", default=False, type=bool)
parser.add_argument("--appliance", type=str, default="Refrigerator")
parser.add_argument("--model", type=str, default="SGPR")
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--n_inducing", type=int, default=512)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--extra", type=str, default="")
parser.add_argument("--n_cpus", type=int, default=1)
parser.add_argument("--n_restarts", type=int, default=1)
args = parser.parse_args()

torch.manual_seed(args.seed)
if "cpu" in args.device:
    torch.set_num_threads(args.n_cpus)

if args.model != 'SVGP' and args.model != 'SGDGP':
    args.batch_size = None

time_arr = []
rmse_arr = []
nlpd_arr = []

for i in range(1):

    if args.mode == "train":
        X_train, y_train = load_data(args)

        # X_train = X_train[:25000, :]
        # # print(X_train)
        # y_train = y_train[:25000]
        print(X_train.shape)
        # TODO: Incorporate model argument below
        model = get_model(X_train, y_train.to(
            args.device), args)
        # Increase raw noise constraint to prevent positive semidefinite error
        # model.mll.likelihood.noise_covar.register_constraint(
        #     "raw_noise", GreaterThan(0.1))
        for param in model.parameters():
            torch.nn.init.normal_(param, 0, 1)
        print(0)
        start_time = time.time()
        model.fit(lr=args.lr, n_epochs=args.epochs, verbose=True,
                  random_state=args.seed, batch_size=args.batch_size, verbose_gap=10)
        end_time = time.time()
        model_name = get_model_name(args, i)
        print(1)
        torch.save(model.state_dict(), os.path.join(
            Config.model_path, model_name))
        time_arr.append((end_time-start_time)/(60*args.n_restarts))
        print("Trained: " + str(i+1))
        # if i == 4:
        #     print("Training Time: " + str(np.mean(np.array(time_arr))) +
        #           " ± " + str(np.std(np.array(time_arr))))
    elif args.mode == "test":
        X_train, y_train, X_test, y_test = load_data(args)
        # TODO: Take model argument
        y_pred_arr = []
        # X_train = X_train[:47000, :]
        # y_train = y_train[:47000]

        model = get_model(X_train, y_train.to(
            args.device), args)
        model_name = get_model_name(args, i)
        model.load_state_dict(torch.load(
            os.path.join(Config.model_path, model_name)))

        with torch.no_grad():
            # TODO: Implement predict batch in skgpytorch
            # TODO: All models should predict a predictive distribution
            if args.batch_predict:
                means, vars = model.predict_batch(
                    X_test, 512, 10)
                rmse_arr.append(torch.square(torch.subtract(
                    means, y_test)).mean().item())
                # if i == 4:
                #     print("RMSE:", np.mean(rmse_arr))
                # rmse = torch.square(torch.subtract(
                #     means, y_test)).mean().item()
                # print("RMSE:", rmse)

            else:
                pred_dist = model.predict(X_test)
                y_pred_arr.append(pred_dist.loc)
                print(pred_dist.mean)
                # TODO: Lets save the results (pred_dist) somehow (even the pickle is fine).
                # rmse_arr.append(mean_squared_error(
                #     pred_dist, y_test, squared=False))
                # nlpd_arr.append(
                #     negative_log_predictive_density(pred_dist, y_test))
                # if i == 9:
                #     print("RMSE: " + str(np.mean(np.array(rmse_arr))) +
                #           " ± " + str(np.std(np.array(rmse_arr))))
                # # print("RMSE:", mean_squared_error(
                # #     pred_dist, y_test, squared=False))
                #     print("NLPD: " + str(np.mean(np.array(nlpd_arr))) +
                #           " ± " + str(np.std(np.array(nlpd_arr))))
            y_pred_arr1 = []
            for i in range(len(y_pred_arr)):
                y_pred_arr1.append(np.array(y_pred_arr[i].cpu()))

            y_mean = np.mean(y_pred_arr1, axis=0)
            y_sigma = np.std(y_pred_arr1, axis=0)
            print("ymean", y_mean)
            y_mean.shape, y_sigma.shape
            scalery = StandardScaler()
            y_ = scalery.fit_transform(y_train.cpu().reshape(-1, 1))
            y_mean = scalery.inverse_transform(
                y_mean.reshape(-1, 1)).squeeze()
            print("trans ymean", y_mean)
            y_sigma = scalery.inverse_transform(
                y_sigma.reshape(-1, 1)).squeeze()
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
            plt.plot(jnp.arange(idx),
                     y_mean[:idx].reshape(-1, 1), label="Predicted")
            for i in range(1, 4):
                plt.fill_between(jnp.arange(idx), y_mean[:idx] - i*y_sigma[:idx], y_mean[:idx] + i*y_sigma[:idx],
                                 color="orange", alpha=(1/(i*3)), label=f"$\mu\pm{i}*\sigma$")
            plt.legend(bbox_to_anchor=(1, 1), fontsize=20)
            plt.ylabel("Power", fontsize=20)
            sns.despine()
            plt.savefig("gp_ref_sgpr_both.png")
    else:
        raise ValueError("Invalid mode")
