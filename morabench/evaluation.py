from morabench.utils import accuracy
from scipy.stats import spearmanr
import numpy as np



def ranking_correction_comp(list1, list2):
    if set(list1) != set(list2) or len(list1) != len(list2):
        raise ValueError("Both lists must have the same unique string elements and same length.")
    name_to_rank = {name: idx for idx, name in enumerate(list1)}
    ranks_list1 = [name_to_rank[name] for name in list1]
    ranks_list2 = [name_to_rank[name] for name in list2]
    for ik, k in enumerate(ranks_list2):
        ranks_list2[ik] = ranks_list2[ik] + 100
    correlation_coefficient, p_value = spearmanr(ranks_list1, ranks_list2)
    return correlation_coefficient


def optimal_gap_comp(list1,list2,test_labels):
    acc_ensemble_result = accuracy(np.array([list1]), test_labels)
    acc_label_result = accuracy(np.array([list2]), test_labels)
    optimal_gap = acc_ensemble_result - acc_label_result
    return optimal_gap
