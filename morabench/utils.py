import copy
import numpy as np
import math
import random
import torch
import pickle


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def read_pkl_file(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data




def normalized_entropy(prob_dist):
    entropy_list = []
    prob_dist = np.squeeze(prob_dist)
    r, c = prob_dist.shape
    for i in range(r):
        prob_dist_tmp = copy.deepcopy(prob_dist[i])
        prob_dist_tmp = np.where(prob_dist_tmp == 0, 1e-9, prob_dist_tmp)
        log_probs = prob_dist_tmp * np.log2(prob_dist_tmp)
        raw_entropy = 0 - np.sum(log_probs, axis=0)
        normalized_entropy = raw_entropy / math.log2(len(log_probs))
        entropy_list.append(normalized_entropy)
    return np.array(entropy_list)


def compute_uncertainty(prob_dist):
    uncertainty_list = []
    prob_dist = np.squeeze(prob_dist)
    r, c = prob_dist.shape
    for i in range(r):
        prob_dist_tmp = copy.deepcopy(prob_dist[i])
        uncertainty = 1 - np.max(prob_dist_tmp)
        uncertainty_list.append(uncertainty)
    return np.array(uncertainty_list)


def compute_margin(prob_dist):
    margin_list = []
    prob_dist = np.squeeze(prob_dist)
    r, c = prob_dist.shape
    for i in range(r):
        prob_dist_tmp = copy.deepcopy(prob_dist[i])
        prob_dist_tmp = np.where(prob_dist_tmp == 0, 1e-9, prob_dist_tmp)
        sorted_probs = np.sort(prob_dist_tmp)
        margin = sorted_probs[-1] - sorted_probs[-2]
        margin_list.append(-margin)
    return np.array(margin_list)





def accuracy(predictions, labels):
    predicted_labels = np.argmax(predictions, axis=2).flatten()
    correct_predictions = np.sum(predicted_labels == labels)
    accuracy = correct_predictions / len(labels)
    return accuracy










