import torch
import sys
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.colors as mcolors
import pandas as pd


EoR_colour = mcolors.LinearSegmentedColormap.from_list('EoR_colour',
             [(0, 'white'),(0.33, 'yellow'),(0.5, 'orange'),(0.68, 'red'),
              (0.83333, 'black'),(0.9, 'blue'),(1, 'cyan')])
plt.register_cmap(cmap=EoR_colour)
plt.rcParams.update({
    'axes.titlesize': 16,     # Title font size
    'axes.labelsize': 16,     # Axes labels font size
    'xtick.labelsize': 16,    # x-tick labels font size
    'ytick.labelsize': 16,    # y-tick labels font size
    'legend.fontsize': 16,    # Legend font size
    'font.size': 16           # Overall font size
})

def read_LC(file: str):
    cone = np.load(file)
    data = cone['image']
    labels = cone['label']
    return data,labels

def plot_LC(ax, data: np.ndarray, resolution: int = 1) -> plt.Figure:
    # Extract the surface points with the given resolution
    x, y, z = data.shape
    # Front surface (x=0)
    surface_x0 = data[0, ::resolution, ::resolution]
    x0, y0, z0 = np.meshgrid([0], np.arange(0, y, resolution), np.arange(0, z, resolution), indexing='ij')
    # Back surface (x=max)
    surface_x1 = data[-1, ::resolution, ::resolution]
    x1, y1, z1 = np.meshgrid([x-1], np.arange(0, y, resolution), np.arange(0, z, resolution), indexing='ij')
    # Left surface (y=0)
    surface_y0 = data[::resolution, 0, ::resolution]
    x2, y2, z2 = np.meshgrid(np.arange(0, x, resolution), [0], np.arange(0, z, resolution), indexing='ij')
    # Right surface (y=max)
    surface_y1 = data[::resolution, -1, ::resolution]
    x3, y3, z3 = np.meshgrid(np.arange(0, x, resolution), [y-1], np.arange(0, z, resolution), indexing='ij')
    # Bottom surface (z=0)
    surface_z0 = data[::resolution, ::resolution, 0]
    x4, y4, z4 = np.meshgrid(np.arange(0, x, resolution), np.arange(0, y, resolution), [0], indexing='ij')
    # Top surface (z=max)
    surface_z1 = data[::resolution, ::resolution, -1]
    x5, y5, z5 = np.meshgrid(np.arange(0, x, resolution), np.arange(0, y, resolution), [z-1], indexing='ij')
    
    # Combine all surface points
    X = np.concatenate([x0.ravel(), x1.ravel(), x2.ravel(), x3.ravel(), x4.ravel(), x5.ravel()])
    Y = np.concatenate([y0.ravel(), y1.ravel(), y2.ravel(), y3.ravel(), y4.ravel(), y5.ravel()])
    Z = np.concatenate([z0.ravel(), z1.ravel(), z2.ravel(), z3.ravel(), z4.ravel(), z5.ravel()])
    values = np.concatenate([surface_x0.ravel(), surface_x1.ravel(), surface_y0.ravel(), surface_y1.ravel(), surface_z0.ravel(), surface_z1.ravel()])
    
    # Plotting
    scatter = ax.scatter(X, Y, Z, c=values, cmap='EoR_colour',marker='s', s=10*resolution, vmin=-180, vmax=40)
    
    # Set zoom and angle view
    elevation_angle = 20  # Adjust as needed
    azimuthal_angle = 150  # Adjust as needed
    ax.invert_zaxis()
    ax.view_init(elev=elevation_angle, azim=azimuthal_angle, vertical_axis='x')
    ax.set_box_aspect([1, 5, 1])
    
    # Removing the axes
    ax.set_axis_off()
    
    return scatter

def shuffle(object):
    shuffled = np.zeros(object.shape)
    shuffled[:,0] = object[:,1]
    shuffled[:,1] = object[:,0]
    shuffled[:,2] = object[:,4]
    shuffled[:,3] = object[:,5]
    shuffled[:,4] = object[:,3]
    shuffled[:,5] = object[:,2]
    return shuffled

def read_samples(file: str):
    samples = np.load(file)['samp']
    labels = np.load(file)['labels']
    labels = shuffle(labels)
    samples=shuffle(samples)
    return samples[::100,:],labels

    
parameters_shuffle=[["OMm",0.2,0.4,r"$\Omega_m$",r"\Omega_m",''],["WDM",0.3,10,r"$m_{WDM}$",r"m_{WDM}",r'$\,\mathrm{[kev]}$'],
            ["Tvir",4,5.3,r"$\log_{10}(T_{vir})$",r"\log_{10}(T_{vir})",r'$\,\mathrm{[K]}$'],["Zeta",10,250,r"$\zeta$",r"\zeta",''],
            ["LX",38,42,r"$\log_{10}(L_X)$",r"log_{10}(L_X)",r'$\,[\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{M}_\odot \,\mathrm{yr}]$'],["E0",100,1500,r"$E_0$",r"E_0",r'$\,\mathrm{[eV]}$']
            ]
lab=[parameters_shuffle[para][3] for para in range(len(parameters_shuffle))]

lc_num = np.arange(30)
lc_paths = ['../mocks/output/run_default.npz']
sample_paths = ['../output/test_posteriors_opt_grid/dist_default.npz']
path_base = '../mocks/output/run'
sample_base = '../output/test_posteriors_opt_grid/dist'
for l in lc_num:
    lc_path = path_base+str(l)+'.npz'
    sample_path = sample_base+str(l)+'.npz'
    if not os.path.exists(lc_path) or not os.path.exists(sample_path):
        continue
    lc_paths.append(lc_path)
    sample_paths.append(sample_path)
lc_paths.append('../mocks/output/run_default.npz')
sample_paths.append('../output/test_posteriors_opt_grid/dist_default.npz')
_,labels_fid = read_samples(sample_paths[0])
    
color='darkred'
h=22
w=22
fig = plt.figure(figsize=(15,15 ))

gs = gridspec.GridSpec(h, w, figure=fig)  # 12 rows, 6 columns grid


ax1 = fig.add_subplot(gs[:14, 3:], projection='3d')

param_axes = [fig.add_subplot(gs[0, 3*i:3*i+2]) for i in range(6)]
for ax in param_axes:
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
scale = 3
# Create pair plot axes in the last 6 rows
pairplot_axes = [[fig.add_subplot(gs[scale*i+(h-scale*6):scale*(i+1)+(h-scale*6),
                                     j*scale:(j+1)*scale]) for j in range(6)] for i in range(6)]

text_x_shift = 20
text_axes_param = fig.add_subplot(gs[0,text_x_shift:])
text_axes_data = fig.add_subplot(gs[6,text_x_shift:])
text_axes_infer = fig.add_subplot(gs[14,text_x_shift:])
line_axes = fig.add_subplot(gs[:,text_x_shift-1:])

for i in range(6):
    for j in range(i + 1, 6):
        pairplot_axes[i][j].remove()
        pairplot_axes[i][j] = None
for row in pairplot_axes:
    for ax in row:
        if ax is not None:
            for spine in ax.spines.values():
                spine.set_linewidth(1.5) 
plt.subplots_adjust(wspace=0, hspace=0)

            
def update_pairplot_axes(data, labels, ground_truth, levels, color='teal',color_cross='darkred'):
    df = pd.DataFrame(data, columns=labels)
    # g = sns.PairGrid(df, corner=True)
    for i in range(len(labels)):
        for j in range(i + 1):
            # if (g.axes[i, j] is not None) & (i != j):
            if j < i:
                pairplot_axes[i][j].clear()
                sns.kdeplot(data=df, x=labels[j], y=labels[i], levels=levels, fill=True, ax=pairplot_axes[i][j],cmap='Blues_d')
                pairplot_axes[i][j].axhline(ground_truth[:,i], color=color_cross, linestyle='--')
                pairplot_axes[i][j].axvline(ground_truth[:,j], color=color_cross, linestyle='--')
            if i == j:
                pairplot_axes[i][j].clear()
                sns.kdeplot(data=df[labels[i]], levels=levels, fill=True, ax=pairplot_axes[i][i])
                pairplot_axes[i][i].axvline(ground_truth[:,i], color=color_cross, linestyle='--')
                
            # Turn off labels and ticks for internal plots
            if i < len(labels)-1:
                pairplot_axes[i][j].xaxis.set_visible(False)
            if j > 0:
                pairplot_axes[i][j].yaxis.set_visible(False)
            pairplot_axes[0][0].yaxis.set_visible(False)
    # g.close()
            
def plot_params(value:np.ndarray,color_index:int=None):
    for i in range(6):
        param_axes[i].set_facecolor('white')
        param_axes[i].clear()
        param_axes[i].axvline(value[i],lw=3,color = color)
        param_axes[i].set_xlim([parameters_shuffle[i][1],parameters_shuffle[i][2]])
        param_axes[i].set_ylim([0,1])
        param_axes[i].set_yticklabels([])
        param_axes[i].set_title(parameters_shuffle[i][3])
        param_axes[i].set_xticks([parameters_shuffle[i][1],parameters_shuffle[i][2]])
        # set background color of plot
        if color_index is not None:
            if i == color_index:
                param_axes[i].set_facecolor((1, 0.41, 0.71, 0.5))



cone,_ = read_LC(lc_paths[0])
data, labels = read_samples(sample_paths[0])

# Initial plots
plot_LC(ax1, cone,resolution=1)
update_pairplot_axes(data, lab, labels, levels=5, color='teal')
plot_params(labels[0])

fs = 30
text_axes_param.text(0, 0.2, 'Parameters', fontsize=fs)
text_axes_param.axis('off')
text_axes_data.text(0, 0.2, 'Data', fontsize=fs)
text_axes_data.axis('off')
text_axes_infer.text(0, 0.2, 'Inference', fontsize=fs)
text_axes_infer.axis('off')

line_axes.axvline(0, lw=2, color='k', linestyle='-')
line_axes.axis('off')
     
# Function to update the plots
def update(frame: int)->plt.Figure:
    ax1.clear()
    cone,_ = read_LC(lc_paths[frame])
    scatter = plot_LC(ax1, cone,resolution=1)
    # Update data for seaborn plot

    new_data, labels = read_samples(sample_paths[frame])
    index = np.where(~np.isclose(labels[0], labels_fid[0], atol=1e-4))[0]
    print(len(index))
    if len(index) == 0:
        index = 4
    if (frame == 0) or (frame == len(lc_paths)-1):
        index = None
    print(index)
    update_pairplot_axes(new_data, lab, labels, levels=5)
    plot_params(labels[0],color_index=index)
    return [scatter]

# # Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0,len(lc_paths)), interval=1000, blit=False)
# Save the animation
ani.save('animation.mp4', writer='ffmpeg', dpi=300)


# Show plot
plt.show()
