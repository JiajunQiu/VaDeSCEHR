"""
Utility functions for model evaluation.
"""
import numpy as np

from lifelines.utils import concordance_index

import sys
import numpy as np
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
import tensorflow as tf
from collections import Counter
from lifelines import KaplanMeierFitter
import itertools
from scipy import stats
from scipy.stats import linregress
import itertools
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import balanced_accuracy_score
sys.path.insert(0, '../')

def compute_joint_probability_distribution(x, y):
    joint_distribution = np.histogram2d(x, y, bins=(len(np.unique(x)), len(np.unique(y))))[0]
    joint_distribution /= joint_distribution.sum()
    return joint_distribution

def compute_mutual_information(joint_distribution):
    x_marginal = joint_distribution.sum(axis=1)
    y_marginal = joint_distribution.sum(axis=0)
    mi = 0
    for i in range(joint_distribution.shape[0]):
        for j in range(joint_distribution.shape[1]):
            if joint_distribution[i, j] > 0:
                mi += joint_distribution[i, j] * np.log2(joint_distribution[i, j] / (x_marginal[i] * y_marginal[j]))
    return mi

def compute_normalized_mutual_information(x, y):
    joint_distribution = compute_joint_probability_distribution(x, y)
    mi = compute_mutual_information(joint_distribution)
    joint_entropy = entropy(joint_distribution.flatten(), base=2)
    nmi = mi / joint_entropy
    return nmi

def calculate_nmi_multi(true_labels,predicted_clusters):
    labels=set(true_labels)
    res_labels=[[] for _ in range(len(labels))]
    # Iterate over each predicted cluster
    for cluster_id in set(predicted_clusters):
        cluster_indices = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
        cluster_labels = [true_labels[i] for i in cluster_indices]

        # Count the occurrences of each true label within the cluster
        label_counts = Counter(cluster_labels)

        for idx,label in enumerate(labels):
            res_labels[idx].append(label_counts[label])

    return compute_normalized_mutual_information(res_labels[0],res_labels[1])


def balanced_cluster_acc(y_true, y_pred):
    """
    Calculate balanced accuracy for clustering.
    
    Balanced accuracy is defined as the arithmetic mean of sensitivity (true positive rate) and specificity (true negative rate).
    
    # Arguments
    y_true: true labels, numpy.array with shape `(n_samples,)`
    y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    
    # Return
    Balanced accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.astype(int).max(), y_true.astype(int).max()) + 1
    w = np.zeros((int(D), (D)), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)
    return np.mean([w[row_ind[i], col_ind[i]]/sum(w[:, col_ind[i]]) for i in range(len(row_ind))])


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.astype(int).max(), y_true.astype(int).max()) + 1
    w = np.zeros((int(D), (D)), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)
    return sum([w[row_ind[i], col_ind[i]] for i in range(len(row_ind))]) * 1.0 / y_pred.size

def find_best_assignment(true_labels,predicted_clusters):
    true_labels=true_labels.astype(np.int64)
    predicted_clusters=predicted_clusters.astype(np.int64)
    num_clusters = max(predicted_clusters)+1
    num_labels = max(true_labels)+1
    assignments = list(itertools.product(range(num_labels), repeat=num_clusters))
    max_nmi = 0.0
#    best_assignment = None

    for assignment in assignments:
        assigned_labels = [assignment[predicted_cluster] for predicted_cluster in  predicted_clusters]
        nmi = normalized_mutual_info_score(true_labels, assigned_labels)
#        nmi = cluster_acc(np.array(true_labels), np.array(assigned_labels))
        if nmi > max_nmi:
            max_nmi = nmi
#            best_assignment = assignment

#    cluster_label_mapping = {cluster: label for cluster, label in enumerate(best_assignment)}
#    return cluster_label_mapping, max_nmi
    return max_nmi


def calculate_purity_minority_med(true_labels,predicted_clusters):
    label_counts = Counter(true_labels)
    minority_label = label_counts.most_common(2)[1][0]
    purity_scores = []
#    weights=[]
    # Iterate over each predicted cluster
    for cluster_id in set(predicted_clusters):
        cluster_indices = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
        cluster_labels = [true_labels[i] for i in cluster_indices]
        if minority_label in cluster_labels:
            # Count the occurrences of each true label within the cluster
            label_counts = Counter(cluster_labels)

            # Calculate purity by dividing the count of the majority label by the total instances
            purity = label_counts[minority_label] / len(cluster_indices)
            purity_scores.extend([purity for _ in range(len(cluster_indices))])
#            weights.append(label_counts[minority_label])

#    return np.average(purity_scores,weights=weights)
    return np.median(purity_scores)

def calculate_purity_minority_top(true_labels,predicted_clusters):
    label_counts = Counter(true_labels)
    minority_label = label_counts.most_common(2)[1][0]
    purity_scores = []
#    weights=[]
    # Iterate over each predicted cluster
    for cluster_id in set(predicted_clusters):
        cluster_indices = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
        cluster_labels = [true_labels[i] for i in cluster_indices]
        if minority_label in cluster_labels:
            # Count the occurrences of each true label within the cluster
            label_counts = Counter(cluster_labels)

            # Calculate purity by dividing the count of the majority label by the total instances
            purity = label_counts[minority_label] / len(cluster_indices)
            purity_scores.append(purity)
#            weights.append(label_counts[minority_label])

#    return np.average(purity_scores,weights=weights)
    return np.max(purity_scores)


def calculate_purity(true_labels,predicted_clusters):
    purity_scores = []
    weights=[]
    # Iterate over each predicted cluster
    for cluster_id in set(predicted_clusters):
        cluster_indices = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
        cluster_labels = [true_labels[i] for i in cluster_indices]

        # Count the occurrences of each true label within the cluster
        label_counts = Counter(cluster_labels)

        # Find the majority true label within the cluster
        majority_label = label_counts.most_common(1)[0][0]

        # Calculate purity by dividing the count of the majority label by the total instances
        purity = label_counts[majority_label] / len(cluster_indices)
        purity_scores.append(purity)
        weights.append(len(cluster_indices))

    return np.average(purity_scores,weights=weights)

def calculate_auc(true_labels,p_c_z):
    predicted_clusters = np.argmax(p_c_z, axis=-1)
    weights=[]
    aucs=[]
    # Iterate over each predicted cluster
    for cluster_id in set(predicted_clusters):
        cluster_indices = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
        cluster_labels = [true_labels[i] for i in cluster_indices]
        cluster_pred = [p_c_z[i,cluster_id] for i in cluster_indices]
        # Count the occurrences of each true label within the cluster
        if len(set(cluster_labels))==1:
            auc=1
        else:
            auc = roc_auc_score(cluster_labels, cluster_pred)
        aucs.append(auc)
        weights.append(len(cluster_indices))

    return np.average(aucs,weights=weights)


'''
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    y_true_unique = np.unique(y_true)
    y_pred_unique = np.unique(y_pred)
    values=[]

    lst = list(range(len(y_pred_unique)-len(y_true_unique)))
    combs = list(itertools.combinations_with_replacement(lst, len(lst)))
    for idx,v in enumerate(combs):
        p = list(range(len(y_true_unique),len(y_pred_unique)))
        map = list(zip(p,v))
        y_pred_tmp = y_pred
        for x in map:
            y_pred_tmp=np.where(y_pred == x[0], x[1] * np.ones_like(y_pred), y_pred)

            D = max(y_pred_tmp.astype(int).max(), y_true.astype(int).max()) + 1
            w = np.zeros((int(D), (D)), dtype=np.int64)
            for i in range(y_pred_tmp.size):
                w[int(y_pred_tmp[i]), int(y_true[i])] += 1
            row_ind, col_ind = linear_assignment(w.max() - w)
            values.append(sum([w[row_ind[i], col_ind[i]] for i in range(len(row_ind))]) * 1.0 / y_pred_tmp.size)
    return max(values)
'''

def accuracy_metric2(inp, p_c_z):
    y_true = inp[:, 2]
    y_pred = tf.math.argmax(p_c_z, axis=-1)
    return tf.numpy_function(cluster_acc, [y_true, y_pred], tf.float64)

def cls_log_los(inp, p_c_z):
    y = inp[:, 2]
    y_pred = tf.math.argmax(p_c_z, axis=-1)
#    return tf.numpy_function(normalized_mutual_info_score, [y, y_pred], tf.float64)
    return tf.numpy_function(log_loss, [y, y_pred], tf.float64)

def cls_acc(inp, y_pred):
    y = inp[:, 2]
    y_pred=tf.squeeze(y_pred)
    y_pred = tf.where(y_pred>0.5,1,0)
#    return tf.numpy_function(normalized_mutual_info_score, [y, y_pred], tf.float64)
    return tf.numpy_function(accuracy_score, [y, y_pred], tf.float64)

def accuracy_metric_multi_med(inp, p_c_z):
    y = inp[:, 2]
    y_pred = tf.math.argmax(p_c_z, axis=-1)
    return tf.numpy_function(calculate_purity_minority_med, [y, y_pred], tf.float64)

def accuracy_metric_multi_top(inp, p_c_z):
    y = inp[:, 2]
    y_pred = tf.math.argmax(p_c_z, axis=-1)
    return tf.numpy_function(calculate_purity_minority_top, [y, y_pred], tf.float64)

def nmi_metric_multi(inp, p_c_z):
    y = inp[:, 2]
    y_pred = tf.math.argmax(p_c_z, axis=-1)
    return tf.numpy_function(calculate_nmi_multi, [y, y_pred], tf.float64)

def auc_metric_multi(inp, p_c_z):
    y = inp[:, 2]
    return tf.numpy_function(calculate_auc, [y,p_c_z], tf.float64)

def nmi_metric(inp, p_c_z):
    y = inp[:, 2]
    y_pred = tf.math.argmax(p_c_z, axis=-1)
    return tf.numpy_function(normalized_mutual_info_score, [y, y_pred], tf.float64)

def accuracy_metric(inp, p_c_z):
    y = inp[:, 2]
    y_pred = tf.math.argmax(p_c_z, axis=-1)
    return tf.numpy_function(cluster_acc, [y, y_pred], tf.float64)
#    return tf.numpy_function(find_best_assignment, [y, y_pred], tf.float64)

def balance_check(inp, p_c_z):
    y_pred = tf.math.argmax(p_c_z, axis=-1)
#    return tf.numpy_function(normalized_mutual_info_score, [y, y_pred], tf.float64)
    return tf.reduce_sum(y_pred)/tf.cast(tf.shape(y_pred)[0], tf.int64) * 100.0

def cindex_metric(inp, risk_scores):
    # Evaluates the concordance index based on provided predicted risk scores, computed using hard clustering
    # assignments.
    t = inp[:, 0]
    d = inp[:, 1]
    risk_scores = tf.squeeze(risk_scores)
    return tf.cond(tf.reduce_any(tf.math.is_nan(risk_scores)),
                   lambda: tf.numpy_function(cindex, [t, d, tf.zeros_like(risk_scores)], tf.float64),
                   lambda: tf.numpy_function(cindex, [t, d, risk_scores], tf.float64))


def cindex(t: np.ndarray, d: np.ndarray, scores_pred: np.ndarray):
    """
    Evaluates concordance index based on the given predicted risk scores.

    :param t: observed time-to-event.
    :param d: labels of the type of even observed. d[i] == 1, if the i-th event is failure (death); d[i] == 0 otherwise.
    :param scores_pred: predicted risk/hazard scores.
    :return: return the concordance index.
    """
    try:
        ci = concordance_index(event_times=t, event_observed=d, predicted_scores=scores_pred)
    except ZeroDivisionError:
        print('Cannot devide by zero.')
        ci = float(0.5)
    return ci


def rae(t_pred, t_true, cens_t):
    # Relative absolute error as implemented by Chapfuwa et al.
    abs_error_i = np.abs(t_pred - t_true)
    pred_great_empirical = t_pred > t_true
    min_rea_i = np.minimum(np.divide(abs_error_i, t_true + 1e-8), 1.0)
    idx_cond = np.logical_and(cens_t, pred_great_empirical)
    min_rea_i[idx_cond] = 0.0

    return np.sum(min_rea_i) / len(t_true)


def calibration(predicted_samples, t, d):
    kmf = KaplanMeierFitter()
    kmf.fit(t, event_observed=d)

    range_quant = np.arange(start=0, stop=1.010, step=0.010)
    t_empirical_range = np.unique(np.sort(np.append(t, [0])))
    km_pred_alive_prob = [kmf.predict(i) for i in t_empirical_range]
    empirical_dead = 1 - np.array(km_pred_alive_prob)

    km_dead_dist, km_var_dist, km_dist_ci = compute_km_dist(predicted_samples, t_empirical_range=t_empirical_range,
                                                            event=d)

    slope, intercept, r_value, p_value, std_err = linregress(x=km_dead_dist, y=empirical_dead)

    return slope


# Bounds
def ci_bounds(surv_t, cumulative_sq_, alpha=0.95):
    # print("surv_t: ", surv_t, "cumulative_sq_: ", cumulative_sq_)
    # This method calculates confidence intervals using the exponential Greenwood formula.
    # See https://www.math.wustl.edu/%7Esawyer/handouts/greenwood.pdf
    # alpha = 0.95
    if surv_t > 0.999:
        surv_t = 1
        cumulative_sq_ = 0
    alpha = 0.95
    constant = 1e-8
    alpha2 = stats.norm.ppf((1. + alpha) / 2.)
    v = np.log(surv_t)
    left_ci = np.log(-v)
    right_ci = alpha2 * np.sqrt(cumulative_sq_) * 1 / v

    c_plus = left_ci + right_ci
    c_neg = left_ci - right_ci

    ci_lower = np.exp(-np.exp(c_plus))
    ci_upper = np.exp(-np.exp(c_neg))

    return [ci_lower, ci_upper]


# Population wise cdf
def compute_km_dist(predicted_samples, t_empirical_range, event):
    km_dead = []
    km_surv = 1

    km_var = []
    km_ci = []
    km_sum = 0

    kernel = []
    e_event = event

    for j in np.arange(len(t_empirical_range)):
        r = t_empirical_range[j]
        low = 0 if j == 0 else t_empirical_range[j - 1]
        area = 0
        censored = 0
        dead = 0
        at_risk = len(predicted_samples)
        count_death = 0
        for i in np.arange(len(predicted_samples)):
            e = e_event[i]
            if len(kernel) != len(predicted_samples):
                kernel_i = stats.gaussian_kde(predicted_samples[i])
                kernel.append(kernel_i)
            else:
                kernel_i = kernel[i]
            at_risk = at_risk - kernel_i.integrate_box_1d(low=0, high=low)

            if e == 1:
                count_death += kernel_i.integrate_box_1d(low=low, high=r)
        if at_risk == 0:
            break
        km_int_surv = 1 - count_death / at_risk
        km_int_sum = count_death / (at_risk * (at_risk - count_death))

        km_surv = km_surv * km_int_surv
        km_sum = km_sum + km_int_sum

        km_ci.append(ci_bounds(cumulative_sq_=km_sum, surv_t=km_surv))

        km_dead.append(1 - km_surv)
        km_var.append(km_surv * km_surv * km_sum)

    return np.array(km_dead), np.array(km_var), np.array(km_ci)



