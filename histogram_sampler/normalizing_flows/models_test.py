import models, transforms
import matplotlib.pyplot as plt
import numpy as np
import normflows as nf
import pandas as pd
import os
import torch

# Prepare data
df = pd.read_parquet("corrected_JZ2_1_file.parquet")
data = df['r21_125_15'].values[0]

pt_cor = torch.Tensor(np.concatenate([entry['pt_uncor'] for entry in data])).unsqueeze(1)
eta_cor = torch.Tensor(np.concatenate([entry['eta_uncor'] for entry in data])).unsqueeze(1)
phi_cor = torch.Tensor(np.concatenate([entry['phi_uncor'] for entry in data])).unsqueeze(1)

pt_eta_corr = torch.cat((pt_cor, eta_cor), dim=1)

hist = np.histogramdd(pt_eta_corr, bins=50)

iter = 100

real_nvp = models.RealNVP(hist)
real_nvp.train(max_iter=iter)
real_nvp_loss = real_nvp.loss_hist

plt.plot(np.arange(iter), real_nvp_loss)
plt.show()

# data_np = pt_eta_corr.numpy()

# Extract x and y components[]
# x = data_np[:, 0]
# y = data_np[:, 1]

# x = pt_cor.numpy().ravel()
# y = eta_cor.numpy().ravel()

# print(x.shape)
# print(y.shape)

# # Compute 2D histogram
# counts, xedges, yedges = np.histogram2d(x, y, bins=20)

# # Compute bin centers
# x_centers = 0.5 * (xedges[1:] + xedges[:-1])
# y_centers = 0.5 * (yedges[1:] + yedges[:-1])
# X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

# # Flatten for bar3d
# xpos = X.ravel()
# ypos = Y.ravel()
# zpos = np.zeros_like(xpos)
# dz = counts.ravel()

# # Width of each bar
# dx = dy = (xedges[1] - xedges[0]) * 0.9

# # Plot 3D bar histogram
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='orange', shade=True)

# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Count')
# plt.title("3D Histogram of 2D Data")
# plt.show()

# plt.scatter(pt_cor, eta_cor, s=2)
# plt.show()




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
