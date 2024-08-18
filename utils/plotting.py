"""
Utility functions for plotting.
"""
import os

import numpy as np

from lifelines import KaplanMeierFitter

import matplotlib.pyplot as plt
from matplotlib import rc

from openTSNE import TSNE as fastTSNE
import umap
import sys

sys.path.insert(0, '../')

CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00','#1F6EC8', '#FF9F20', '#2D8F2A', '#F759A7', '#A6504A', '#9C4E83', '#797979', '#E4323C', '#DEDE14','#4F8ED8', '#FF5F00', '#6DBF6A', '#F7A9D7', '#A67658', '#984EBD', '#B3B3B3', '#E4000C', '#DEDE46']
GRAY_COLOR_CYCLE = ['black', 'dimgray', 'darkgray', 'gainsboro', 'whitesmoke']
LINE_TYPES = ['solid', 'dashed', 'dashdot', 'dotted', 'dashed']
MARKER_STYLES = ['', '', '', '', '']
DASH_STYLES = [[], [4, 4], [4, 1], [1, 1, 1], [2, 1, 2]]
 

def plotting_setup(font_size=12):
    # plot settings
    plt.style.use("seaborn-colorblind")
    plt.rcParams['font.size'] = font_size
    rc('text', usetex=False)


def plot_overall_kaplan_meier(t, d, dir=None):
    kmf = KaplanMeierFitter()
    kmf.fit(t, d, label="Overall KM estimate")
    kmf.plot(ci_show=True)
    if dir is not None:
        plt.savefig(fname=os.path.join(dir, "km_plot.png"), dpi=300, pad_inches=0.2)
    plt.show()


def plot_group_kaplan_meier(t, d, c, dir=None, experiment_name='',selected_samples=[]):
    if len(selected_samples)>0:
        fig = plt.figure()
        labels = np.unique(c)
        for l in labels:
            kmf = KaplanMeierFitter()
            kmf.fit(t[(c == l) & (selected_samples==1)], d[(c == l) & (selected_samples==1)], label="Cluster " + str(int(l + 1)))
            kmf.plot(ci_show=True, color=CB_COLOR_CYCLE[int(l)])
        plt.xlabel("Time (years)")
        plt.ylabel("Risk free Probability")
        if dir is not None:
            plt.savefig(fname=os.path.join(dir, "km_group_plot_" + experiment_name +".png"), dpi=300, bbox_inches="tight")
        else:
            plt.show()
    else:
        fig = plt.figure()
        labels = np.unique(c)
        for l in labels:
            kmf = KaplanMeierFitter()
            kmf.fit(t[c == l], d[c == l], label="Cluster " + str(int(l + 1)))
            kmf.plot(ci_show=True, color=CB_COLOR_CYCLE[int(l)])
        plt.xlabel("Time (years)")
        plt.ylabel("Risk free Probability")
        if dir is not None:
            plt.savefig(fname=os.path.join(dir, "km_group_plot_" + experiment_name +".png"), dpi=300, bbox_inches="tight")
        else:
            plt.show()        

def plot_bigroup_kaplan_meier(t, d, c, c_, dir=None, postfix=None, legend=False, legend_outside=False,experiment_name=''):
    fig = plt.figure()

    # Plot true clusters
    labels = np.unique(c)
    for l in labels:
        kmf = KaplanMeierFitter()
        if legend:
            kmf.fit(t[c == l], d[c == l], label="Cluster " + str(int(l + 1)))
        else:
            kmf.fit(t[c == l], d[c == l])
        kmf.plot(ci_show=True, alpha=0.75, color=CB_COLOR_CYCLE[int(l)], linewidth=5)

    # Plot assigned clusters
    labels = np.unique(c_)
    for l in labels:
        kmf = KaplanMeierFitter()
        if legend:
            kmf.fit(t[c_ == l], d[c_ == l], label="Ass. cluster " + str(int(l + 1)))
        else:
            kmf.fit(t[c_ == l], d[c_ == l])
        kmf.plot(ci_show=True, color='black', alpha=0.25, linestyle=LINE_TYPES[int(l)], dashes=DASH_STYLES[int(l)],
                 linewidth=5)

    plt.xlabel("Time (years)")
    plt.ylabel("Risk free Probability")

    if legend:
        if legend_outside:
            leg = plt.legend(loc='upper right', frameon=False, bbox_to_anchor=(-0.15, 1))
        else:
            leg = plt.legend(loc='lower right', frameon=False)
    else:
        leg = plt.legend('', frameon=False)

    if dir is not None:
        plt.savefig(fname=os.path.join(dir, "km_group_plot_" + experiment_name +".png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_dataset(X, t, d, c, font_size=12, seed=42, dir=None, postfix=None):
    plotting_setup(font_size=font_size)

    plot_group_kaplan_meier(t=t, d=d, c=c, dir=dir)

    if X.shape[0] > 10000:
        inds = np.random.choice(a=np.arange(0, X.shape[0]), size=(10000, ))
        c_ = c[inds]
        X_ = X[inds]
    else:
        c_ = c
        X_ = X

    X_embedded = fastTSNE(n_components=2, n_jobs=8, random_state=seed).fit(X_)

    fig = plt.figure()
    for l in np.unique(c_):
        plt.scatter(X_embedded[c_ == l, 0], X_embedded[c_ == l, 1], s=1.5, c=CB_COLOR_CYCLE[int(l)],
                    label=("Cluster " + str(int(l + 1))))
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(markerscale=3.0)
    if dir is not None:
        fname = 'tsne'
        if postfix is not None:
            fname += '_' + postfix
        fname += '.png'
        plt.savefig(fname=os.path.join(dir, fname), dpi=300)
    else:
        plt.show()


def plot_tsne_by_cluster(X, c, font_size=12, seed=42, dir=None, postfix=None,selected_samples=[]):
    np.random.seed(seed)

    plotting_setup(font_size=font_size)

    if X.shape[0] > 10000:
        inds = np.random.choice(a=np.arange(0, X.shape[0]), size=(10000,))
        c_ = c[inds]
        X_ = X[inds]
    else:
        c_ = c
        X_ = X

    X_embedded = fastTSNE(n_components=2, perplexity=50,n_jobs=8, random_state=seed).fit(X_)

    if len(selected_samples)>0:
        fig = plt.figure()
        for l in np.unique(c_):

            plt.scatter(X_embedded[(c_ == l) & (selected_samples==1), 0], X_embedded[(c_ == l) & (selected_samples==1), 1], s=1.5, c=CB_COLOR_CYCLE[int(l)],
                        label=("Cluster " + str(int(l + 1))))
        plt.xlabel(r'$t$-SNE Dimension 1')
        plt.ylabel(r'$t$-SNE Dimension 2')
        plt.legend(loc='lower right',markerscale=3.0)
        if dir is not None:
            fname = 'tsne_vs_c'
            if postfix is not None:
                fname += '_' + postfix
            fname += '.png'
            plt.savefig(fname=os.path.join(dir, fname), dpi=300)
        else:
            plt.show()
    else:
        fig = plt.figure()
        for l in np.unique(c_):
            plt.scatter(X_embedded[c_ == l, 0], X_embedded[c_ == l, 1], s=1.5, c=CB_COLOR_CYCLE[int(l)],
                        label=("Cluster " + str(int(l + 1))))
        plt.xlabel(r'$t$-SNE Dimension 1')
        plt.ylabel(r'$t$-SNE Dimension 2')
        plt.legend(loc='lower right',markerscale=3.0)
        if dir is not None:
            fname = 'tsne_vs_c'
            if postfix is not None:
                fname += '_' + postfix
            fname += '.png'
            plt.savefig(fname=os.path.join(dir, fname), dpi=300)
        else:
            plt.show()
'''
def plot_umap_by_cluster(X, c, font_size=12, seed=42, dir=None, postfix=None):
    np.random.seed(seed)

    plotting_setup(font_size=font_size)

    if X.shape[0] > 10000:
        inds = np.random.choice(a=np.arange(0, X.shape[0]), size=(10000,))
        c_ = c[inds]
        X_ = X[inds]
    else:
        c_ = c
        X_ = X

    X_embedded = umap.UMAP(min_dist=0.9).fit_transform(X_)

    fig = plt.figure()
    for l in np.unique(c_):
        plt.scatter(X_embedded[c_ == l, 0], X_embedded[c_ == l, 1], s=1.5, c=CB_COLOR_CYCLE[int(l)],
                    label=("Cluster " + str(int(l + 1))))
    plt.xlabel(r'UMAP Dimension 1')
    plt.ylabel(r'UMAP Dimension 2')
    plt.legend(loc='upper left',markerscale=3.0)
    if dir is not None:
        fname = 'umap_vs_c'
        if postfix is not None:
            fname += '_' + postfix
        fname += '.png'
        plt.savefig(fname=os.path.join(dir, fname), dpi=300)
    else:
        plt.show()

def plot_tsne_by_survival(X, t, d, font_size=16, seed=42, dir=None, postfix=None, plot_censored=True):
    np.random.seed(seed)

    plotting_setup(font_size=font_size)

    if X.shape[0] > 10000:
        inds = np.random.choice(a=np.arange(0, X.shape[0]), size=(10000,))
        t_ = t[inds]
        d_ = d[inds]
        X_ = X[inds]
    else:
        t_ = t
        d_ = d
        X_ = X

    X_embedded = fastTSNE(n_components=2, n_jobs=8, random_state=seed).fit(X_)

    fig = plt.figure()
    plt.scatter(X_embedded[d_ == 1, 0], X_embedded[d_ == 1, 1], s=1.5, c=np.log(t_[d_ == 1]), cmap='cividis', alpha=0.5)
    if plot_censored:
        plt.scatter(X_embedded[d_ == 0, 0], X_embedded[d_ == 0, 1], s=1.5, c=np.log(t_[d_ == 0]), cmap='cividis',
                    alpha=0.5, marker='s')
    clb = plt.colorbar()
    clb.ax.set_title(r'$\log(T)$')
    plt.xlabel(r'$t$-SNE Dimension 1')
    plt.ylabel(r'$t$-SNE Dimension 2')
    plt.axis('off')
    if dir is not None:
        fname = 'tsne_vs_t'
        if postfix is not None:
            fname += '_' + postfix
        fname += '.png'
        plt.savefig(fname=os.path.join(dir, fname), dpi=300)
    else:
        plt.show()
'''
'''
from scipy.spatial.distance import jensenshannon
def js_divergence(p, q):
    return jensenshannon(p, q) ** 2

def plot_umap_by_cluster(X, c, font_size=12, seed=42, dir=None, postfix=None):
    np.random.seed(seed)

    plotting_setup(font_size=font_size)

    if X.shape[0] > 10000:
        inds = np.random.choice(a=np.arange(0, X.shape[0]), size=(10000,))
        c_ = c[inds]
        X_ = X[inds]
    else:
        c_ = c
        X_ = X

    num_samples=X_.shape[0]
    js_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            if i != j:
                js_matrix[i, j] = js_divergence(X_[i], X_[j])
            else:
                pass
    js_matrix = np.nan_to_num(js_matrix)
'''
from scipy.spatial.distance import jensenshannon
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

def js_divergence(p, q):
    return jensenshannon(p, q) ** 2

def calculate_js_matrix_row(args):
    i, num_samples, X_ = args
    return [js_divergence(X_[i], X_[j]) if i != j else 0 for j in range(num_samples)]

def plot_umap_by_cluster(X, c, font_size=12, seed=42, dir=None, postfix=None):
    np.random.seed(seed)

    plotting_setup(font_size=font_size)

    if X.shape[0] > 10000:
        inds = np.random.choice(a=np.arange(0, X.shape[0]), size=(10000,))
        c_ = c[inds]
        X_ = X[inds]
    else:
        c_ = c
        X_ = X

    num_samples = X_.shape[0]

    with Pool() as p:
        args = [(i, num_samples, X_) for i in range(num_samples)]
        js_matrix = np.array(list(tqdm(p.imap(calculate_js_matrix_row, args), total=num_samples)))

    js_matrix = np.nan_to_num(js_matrix)

    c = np.nan_to_num(c)
    # Check if there are any infinite values in js_matrix and c
    if np.any(np.isinf(js_matrix)) or np.any(np.isinf(c)):
        print("Input contains infinite values.")
        # Handle infinite values here

    # Check if there are any values too large for dtype('float64') in js_matrix and c
    if np.any(js_matrix > np.finfo(np.float64).max) or np.any(c > np.finfo(np.float64).max):
        print("Input contains values too large for dtype('float64').")
        # Handle large values here

    from sklearn import metrics
    sil_score = metrics.silhouette_score(js_matrix, c,metric="precomputed")
    print(f'Silhouette Score: {sil_score}')
    X_embedded = umap.UMAP(n_neighbors=5,min_dist=0.99,spread=30.0,metric='precomputed').fit_transform(js_matrix)

    fig = plt.figure()
    for l in np.unique(c_):
        plt.scatter(X_embedded[c_ == l, 0], X_embedded[c_ == l, 1], s=1.5, c=CB_COLOR_CYCLE[int(l)],
                    label=("Cluster " + str(int(l + 1))))
    plt.xlabel(r'UMAP Dimension 1')
    plt.ylabel(r'UMAP Dimension 2')
    plt.legend(loc='upper left',markerscale=3.0)
    if dir is not None:
        fname = 'umap_vs_c'
        if postfix is not None:
            fname += '_' + postfix
        fname += '.png'
        plt.savefig(fname=os.path.join(dir, fname), dpi=300)
    else:
        plt.show()


def plot_elbow(ks, avg, sd, xlab, ylab, dir=None):
    plotting_setup(16)
    plt.errorbar(ks, avg, yerr=sd, color=CB_COLOR_CYCLE[0], ecolor=CB_COLOR_CYCLE[0], barsabove=True,  marker='D')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if dir is not None:
        plt.savefig(fname=os.path.join(dir, "elbow_plot.png"), dpi=300, bbox_inches="tight")
    plt.show()
