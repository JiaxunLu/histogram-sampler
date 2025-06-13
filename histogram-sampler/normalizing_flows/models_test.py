import models
import matplotlib.pyplot as plt
import numpy as np
import normflows as nf
import pandas as pd
import os
import torch

# mean = np.array([10,5])
# cov = np.array([[7,0],[0,5]])
# data = 4 * np.random.multivariate_normal(mean, cov, 1000)

# d_data = nf.distributions.target.CircularGaussianMixture().sample(1000)
# print(data)
# data = nf.distributions.target.TwoMoons().sample(1000)
# data = nf.distributions.target.RingMixture().sample(1000)

# plt.scatter(data[:,0], data[:,1])
# plt.show()

# np.random.seed(19680801)

# Prepare data
df = pd.read_parquet("corrected_JZ2_1_file.parquet")
data = df['r21_125_15'].values[0]

pt_cor = torch.Tensor(np.concatenate([entry['pt_uncor'] for entry in data])).unsqueeze(1)
eta_cor = torch.Tensor(np.concatenate([entry['eta_uncor'] for entry in data])).unsqueeze(1)
phi_cor = torch.Tensor(np.concatenate([entry['phi_uncor'] for entry in data])).unsqueeze(1)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig_loss, axes_loss = plt.subplots(1, 3, figsize=(15, 8))

pt_histogram = np.histogramdd(pt_cor / 1000, bins=50)
eta_histogram = np.histogramdd(eta_cor, bins=50)
phi_histogram = np.histogramdd(phi_cor, bins=50)

pt_spline = models.AutoregressiveRationalQuadraticSpline(hist=pt_histogram)
pt_spline.train(max_iter=50, lr=0.0002)
pt_trained_data = pt_spline.sample(100000)

eta_spline = models.AutoregressiveRationalQuadraticSpline(hist=eta_histogram)
eta_spline.train(max_iter=50, lr=0.0002)
eta_trained_data = eta_spline.sample(100000)

phi_spline = models.AutoregressiveRationalQuadraticSpline(hist=phi_histogram)
phi_spline.train(max_iter=50, lr=0.0002)
phi_trained_data = phi_spline.sample(100000)

axes_loss[0].plot(np.arange(50), pt_spline.loss_hist)
axes_loss[0].set_title('pt_cor forward KLD loss vs. epochs')

axes_loss[1].plot(np.arange(50), eta_spline.loss_hist)
axes_loss[1].set_title('eta_cor forward KLD loss vs. epochs')

axes_loss[2].plot(np.arange(50), phi_spline.loss_hist)
axes_loss[2].set_title('phi_cor forward KLD loss vs. epochs')

axes[0, 0].hist(pt_cor / 1000, bins=50, histtype="step", color="orange", range=(0, 200))
axes[0, 0].set_title('pt_cor')

axes[0, 1].hist(eta_cor, bins=50, histtype="step", color="orange", range=(-5, 5))
axes[0, 1].set_title('eta_cor')

axes[0, 2].hist(phi_cor, bins=50, histtype="step", color="orange", range=(-4, 4))
axes[0, 2].set_title('phi_cor')

axes[1, 0].hist(pt_trained_data, bins=50, histtype="step", color="blue", range=(0, 200))
axes[1, 0].set_title('pt_cor trained, 50 epochs')

axes[1, 1].hist(eta_trained_data, bins=50, histtype="step", color="blue", range=(-5, 5))
axes[1, 1].set_title('eta_cor trained, 50 epochs')

axes[1, 2].hist(phi_trained_data, bins=50, histtype="step", color="blue", range=(-4, 4))
axes[1, 2].set_title('phi_cor trained, 50 epochs')

plt.show()




# print("Current working directory:", os.getcwd())

# ----------------
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# x, y = np.transpose(data)
# histogram = np.histogramdd(data, bins=10)
# hist, bins = histogram

# Construct arrays for the anchor positions of the 16 bars.
# xpos, ypos = np.meshgrid(bins[0][:-1] + 0.25, bins[1][:-1] + 0.25, indexing="ij")
# xpos = xpos.ravel()
# ypos = ypos.ravel()
# zpos = 0

# # Construct arrays with the dimensions for the 16 bars.
# dx = dy = 0.5 * np.ones_like(zpos)
# dz = hist.ravel()

# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

# plt.show()

# real_nvp = models.real_nvp(hist=histogram)
# real_nvp.train(num_layers=32, max_iter=2000)
# trained_data = real_nvp.sample(1000)[0].detach().numpy()

# spline = models.spline(hist=histogram)
# spline.train(num_layers=32, max_iter=1000)
# trained_data = spline.sample(1000)[0].detach().numpy()

# fig, axes = plt.subplots(nrows=1, ncols=2)
# axes[0].scatter(data[:,0], data[:,1])
# axes[1].scatter(trained_data[:,0], trained_data[:,1])

# axes[0].set_title("target distribution")
# axes[1].set_title("trained distribution")

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 10))
# plt.plot(spline.loss_hist, label='loss')
# plt.legend()
# plt.show()
