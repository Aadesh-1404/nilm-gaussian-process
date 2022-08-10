from utilities import plot, fits, gmm, errors, predict, preprocess
import os
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
from utilities.fits import fit
import tinygp
from tinygp import kernels, GaussianProcess
import tensorflow_probability.substrates.jax as tfp
from datasets.dataset_load import dataset_loader
import jaxopt
os.chdir("../")


# from utilities import plot

dist = tfp.distributions

train = {1: {
    'start_time': "2011-04-21",
    'end_time': "2011-05-21"
},
    3: {
    'start_time': "2011-04-19",
    'end_time': "2011-05-22"
},
    2: {
    'start_time': "2011-04-21",
    'end_time': "2011-05-21"
},
    5: {
    'start_time': "2011-04-22",
    'end_time': "2011-06-01"
}
}
test = {6: {
    'start_time': "2011-05-25",
    'end_time': "2011-06-13"
}}
appliances = ["Refrigerator"]

x_train, y_train, x_test, y_test, x_test_timestamp, scaler_x, scaler_y = dataset_loader(
    appliances, train, test)

dist = tfp.distributions


def build_gp(theta_, x):
    # x = x/(jnp.exp(theta_["len_scale"]))
    # kernel1 = (jnp.exp(theta_["varf_"]))*kernels.ExpSquared(scale=jnp.exp(theta_["len_scale_"]))
    kernel2 = (jnp.exp(theta_["varf"]))*kernels.ExpSineSquared(
        scale=jnp.exp(theta_["len_scale"]), gamma=jnp.exp(theta_["gamma"]))
    kernel = kernel2
    return GaussianProcess(kernel, x, diag=(jnp.exp(theta_["vary"])))

# def build_gp(theta_, x):
#     # x = x/(jnp.exp(theta_["len_scale"]))
#     kernel = (jnp.exp(theta_["varf"]))*kernels.ExpSineSquared(scale=jnp.exp(theta_["len_scale"]), gamma=jnp.exp(theta_["gamma"]))
#     return GaussianProcess(kernel, x, diag=(jnp.exp(theta_["vary"])))


def NLL(theta_, x, y_):
    gp = build_gp(theta_, x)
    return -gp.log_probability(y_)


theta_init = {
    "varf": jnp.log(1.0),
    # "varf_": jnp.log(1.0),
    "vary": jnp.log(1.0),
    "len_scale": jnp.log(1.0),
    # "len_scale_": jnp.log(1.0),
    "gamma": jnp.log(1.0)}


obj = jax.jit(jax.value_and_grad(NLL))


print(
    f"Initial negative log likelihood: {obj(theta_init, x_train, y_train)[0]}")
print(
    f"Gradient of the negative log likelihood, wrt the parameters:\n{obj(theta_init,x_train, y_train)[1]}")

solver = jaxopt.ScipyMinimize(fun=NLL, method='L-BFGS-B')
# print(X.shape, y.shape)
soln = solver.run(theta_init, x_train, y_train)
print(f"Final negative log likelihood: {soln.state.fun_val}")

print(soln.params)


gp = build_gp(soln.params, x_train)
cond_gp = gp.condition(y_train, x_test).gp
mu, var = cond_gp.loc, cond_gp.variance
print(mu.shape, var.shape)
mean = scaler_y.inverse_transform(mu.reshape(-1, 1)).squeeze()
sigma = scaler_y.scale_*jnp.sqrt(var)
print(mean.shape, sigma.shape)


def NLL(mean, sigma, y):
    def loss_fn(mean, sigma, y):
        d = dist.Normal(loc=mean, scale=sigma)
        return -d.log_prob(y)
    return jnp.mean(jax.vmap(loss_fn, in_axes=(0, 0, 0))(mean, sigma, y))


print(NLL(mean, sigma, scaler_y.inverse_transform(y_test.reshape(-1, 1))))

idx = 300
plt.figure(figsize=(10, 6))
plt.plot(jnp.arange(idx), scaler_y.inverse_transform(
    y_test.reshape(-1, 1)).reshape(-1)[:idx], label="Refrigerator", color="green")
plt.plot(jnp.arange(idx), mean[:idx].reshape(-1, 1), label="Predicted")
for i in range(1, 4):
    plt.fill_between(jnp.arange(idx), mean[:idx] - i*sigma[:idx], mean[:idx] + i*sigma[:idx],
                     color="orange", alpha=(1/(i*3)), label=f"$\mu\pm{i}*\sigma$")
plt.legend(bbox_to_anchor=(1, 1), fontsize=20)
plt.ylabel("Power", fontsize=20)
sns.despine()
plt.savefig("gp_ref_full.png")

idx = 3000
plt.figure(figsize=(10, 6))
plt.plot(jnp.arange(idx), scaler_y.inverse_transform(
    y_test.reshape(-1, 1)[:idx]), label="Dish Washer")
plt.plot(jnp.arange(idx), scaler_y.inverse_transform(
    mu.reshape(-1, 1)[:idx]), label="Predicted")
plt.legend(bbox_to_anchor=(1, 1), fontsize=20)
sns.despine()
plt.savefig("gp_ref_full_.png", bbox_inches="tight")
