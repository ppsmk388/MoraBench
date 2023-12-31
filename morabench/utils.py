import math
import torch
import pickle
from morabench.conifg import root_path
import copy
import numpy as np
import random
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


def generate_splits(split_num, dataset_name):
    result_save_path = f'{root_path}/Extracting_information_data/{dataset_name}'
    dataset_matrix_set = read_pkl_file(f'{result_save_path}/result.pkl')
    val_num = len(dataset_matrix_set[dataset_name]['labels'])

    val_indices_list = {}
    test_indices_list = {}
    val_indices_list[dataset_name] = []
    test_indices_list[dataset_name] = []

    def split_data(num_data_points):
        indices = np.random.permutation(num_data_points)
        num_train = int(num_data_points * 0.2)
        val_indices = indices[:num_train]
        test_indices = indices[num_train:]
        return val_indices, test_indices

    for i in range(split_num):
        val_indices, test_indices = split_data(val_num)
        val_indices_list[dataset_name].append(val_indices)
        test_indices_list[dataset_name].append(test_indices)

    return val_indices_list, test_indices_list



# LEMR


def gen_ensemble_matrix(ensemble_dict):
    K = len(ensemble_dict)
    N, M = ensemble_dict[list(ensemble_dict.keys())[0]].shape
    ensemble_matrix = np.zeros((K, N, M))
    for i, S_key in enumerate(ensemble_dict.keys()):
        ensemble_matrix[i] = ensemble_dict[S_key]
    return ensemble_matrix


def max_index_matrix(mat):
    A, B = mat.shape
    max_index = np.argmax(mat, axis=1)
    result = max_index.reshape(A, 1)
    return result.T

def gen_ensemble_result_from_matrix(ensemble_matrix):
    num_models = len(ensemble_matrix)
    ensemble_result_distribution = np.sum(ensemble_matrix, axis=0) / num_models
    ensemble_result = max_index_matrix(ensemble_result_distribution)
    ensemble_result_distribution = np.reshape(ensemble_result_distribution, (1, ensemble_result_distribution.shape[0], ensemble_result_distribution.shape[1]))
    return ensemble_result, ensemble_result_distribution





def calculate_ensemble_label(dict_dict):
    total_matrix = None
    for key, value in dict_dict.items():
        if total_matrix is None:
            total_matrix = value
        else:
            total_matrix += value
    n = len(dict_dict)
    average_matrix = total_matrix / n
    return average_matrix


def generate_hard_ensemble_result(matrix):
    result = np.zeros_like(matrix)
    max_indices = np.argmax(matrix, axis=1)
    result[np.arange(result.shape[0]), max_indices] = 1
    return result



def max_k_indices_except(arr, k, sorted_indices):
    mask = np.ones(arr.shape, dtype=bool)
    mask[sorted_indices] = False
    sub_arr = arr[mask]
    indices = np.argpartition(sub_arr, -k)[-k:]
    sorted_sub_indices = indices[np.argsort(sub_arr[indices])][::-1]
    original_indices = np.arange(arr.shape[0])[mask][sorted_sub_indices]
    return original_indices



def max_k_indices(arr, k):
    indices = np.argpartition(arr, -k)[-k:]
    sorted_indices = indices[np.argsort(arr[indices])][::-1]
    return sorted_indices

def Gt_index_update(array1, array2):
    merged_array = np.concatenate((array1, array2)).astype(int)
    unique_array, _ = np.unique(merged_array, return_counts=True)
    return unique_array.astype(int)

def doing_few_shot_change(arr, labels, index_list):
    arr = np.squeeze(arr)
    for ind in index_list:
        arr[ind] = np.zeros_like(arr[ind])
        arr[ind][labels[ind]] = 1
    def flatten_matrix(matrix):
        return matrix[np.newaxis, :]
    arr = flatten_matrix(arr)
    return arr



def filter_and_preserve_labels(result_data_save_dict, model_acc_dict_filter):
    filtered_dict = {}
    filtered_dict.update({key: value for key, value in result_data_save_dict.items() if key in model_acc_dict_filter})
    return filtered_dict



def accuracy_for_one_model(predict_label, pseudo_label):
    predicted_classes = np.argmax(predict_label, axis=1)
    true_classes = np.argmax(pseudo_label, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    return accuracy


def calculate_accuracy(dict_dict, pseudo_label):
    accuracy_dict = {}
    for key, predict_label in dict_dict.items():
        accuracy = accuracy_for_one_model(copy.deepcopy(predict_label), copy.deepcopy(pseudo_label))
        accuracy_dict[key] = accuracy
    sorted_items = sorted(accuracy_dict.items(), key=lambda x: x[1])
    accuracy_dict = {k: v for k, v in sorted_items}
    return accuracy_dict


def outlier_model_remove(result_dict, threshold):
    data = list(result_dict.values())
    data.sort()
    while True:
        median = np.median(data)
        mad = np.median([np.abs(x - median) for x in data])
        if mad == 0:
            break
        modified_z_scores = [0.6745 * (x - median) / mad for x in data]

        outliers = [data[i] for i, z in enumerate(modified_z_scores) if z < threshold]
        if len(outliers) != 0:
            for l in range(len(outliers)):
                data.pop(0)
        else:
            break
    result_dict_tmp = copy.deepcopy(result_dict)
    for key, value in result_dict.items():
        if value not in data:
            result_dict_tmp.pop(key, None)
    if len(result_dict) == len(result_dict_tmp):
        break_P = True
    else:
        break_P = False
    return result_dict_tmp, break_P






def Pseudo_label_Generation(model_hub,Model_result_dict):
    pseudo_label = calculate_ensemble_label(copy.deepcopy(model_hub))
    model_hub = copy.deepcopy(Model_result_dict)
    return pseudo_label, model_hub





def Active_Label_Acquisition(ensemble_result_distribution, update_budget, labeling_data_index_list,Uncertainty_sampling_strategy,labeling_budget_max_number,labels):
    if Uncertainty_sampling_strategy == 'Entropy':
        Few_shot_Soft_ensemble_result_distribution = normalized_entropy(copy.deepcopy(ensemble_result_distribution))
    elif Uncertainty_sampling_strategy == 'Uncertainty':
        Few_shot_Soft_ensemble_result_distribution = compute_uncertainty(copy.deepcopy(ensemble_result_distribution))
    elif Uncertainty_sampling_strategy == 'Margin':
        Few_shot_Soft_ensemble_result_distribution = compute_margin(copy.deepcopy(ensemble_result_distribution))
    N = update_budget
    if N + len(labeling_data_index_list) > labeling_budget_max_number:
        N = labeling_budget_max_number - len(labeling_data_index_list)
    Gt_index = max_k_indices(Few_shot_Soft_ensemble_result_distribution, N)
    length_labeling_data_index_list_origin = len(labeling_data_index_list)
    Gt_index = Gt_index_update(array1=np.array(labeling_data_index_list), array2=Gt_index)
    length_labeling_data_index_list_change = len(Gt_index)
    if (length_labeling_data_index_list_origin == length_labeling_data_index_list_change) and (len(labeling_data_index_list) != 0):
        Gt_index_compensation = max_k_indices_except(arr=copy.deepcopy(Few_shot_Soft_ensemble_result_distribution), k=N, sorted_indices=copy.deepcopy(Gt_index))
        Gt_index = Gt_index_update(array1=copy.deepcopy(Gt_index_compensation), array2=copy.deepcopy(Gt_index))
    if (length_labeling_data_index_list_change - length_labeling_data_index_list_origin < N) and (length_labeling_data_index_list_change - length_labeling_data_index_list_origin != 0):

        Gt_index_compensation = max_k_indices_except(arr=copy.deepcopy(Few_shot_Soft_ensemble_result_distribution),
                                                     k=N - length_labeling_data_index_list_change + length_labeling_data_index_list_origin,
                                                     sorted_indices=copy.deepcopy(Gt_index))
        Gt_index = Gt_index_update(array1=copy.deepcopy(Gt_index_compensation), array2=copy.deepcopy(Gt_index))
    ensemble_result_distribution = doing_few_shot_change(arr=copy.deepcopy(ensemble_result_distribution), labels=labels, index_list=Gt_index)
    return np.squeeze(ensemble_result_distribution), Gt_index




def Model_Committee_Selection(labeling_data_index_list,Gt_index,model_hub,update_pseudo_label,threshold):
    merged_array = np.concatenate((np.array(labeling_data_index_list), copy.deepcopy(Gt_index))).astype(int)
    labeling_data_index_list, _ = np.unique(merged_array, return_counts=True)
    labeling_data_index_list = labeling_data_index_list.tolist()
    model_hub_acc_dict = calculate_accuracy(copy.deepcopy(model_hub), copy.deepcopy(update_pseudo_label))
    model_acc_dict_filter, break_P = outlier_model_remove(copy.deepcopy(model_hub_acc_dict),
                                                          threshold=threshold,
                                                          )
    model_hub = filter_and_preserve_labels(copy.deepcopy(model_hub), copy.deepcopy(model_acc_dict_filter))
    return labeling_data_index_list, model_hub


def pesudo_gen(labels, Model_result_dict, labeling_budget_ratio, Uncertainty_sampling_strategy='Entropy',threshold=-2.8):
        _, first_value = list(Model_result_dict.items())[0]
        all_data_number, _ = first_value.shape
        labeling_budget_max_number = int(labeling_budget_ratio * all_data_number)
        batch_size_labeling_num = int(labeling_budget_max_number / 10)
        labeling_data_index_list = []
        model_hub = copy.deepcopy(Model_result_dict)
        while len(labeling_data_index_list) < labeling_budget_max_number:
            ##################################################################################################################
            # Step 1: Pseudo_label_Generation
            pseudo_label, model_hub = Pseudo_label_Generation(model_hub=model_hub,
                                                              Model_result_dict=Model_result_dict)
            ##################################################################################################################
            # Step 2: Active_Label_Acquisition
            update_pseudo_label, Gt_index = Active_Label_Acquisition(ensemble_result_distribution=copy.deepcopy(pseudo_label),
                                                                   update_budget=batch_size_labeling_num,
                                                                   labeling_data_index_list=labeling_data_index_list,
                                                                   Uncertainty_sampling_strategy=Uncertainty_sampling_strategy,
                                                                   labeling_budget_max_number=labeling_budget_max_number,
                                                                   labels=labels)
            ##################################################################################################################
            # Step 3: Model_Committee_Selection
            labeling_data_index_list, model_hub = Model_Committee_Selection(labeling_data_index_list=labeling_data_index_list,
                                                                            Gt_index=Gt_index,
                                                                            model_hub=model_hub,
                                                                            update_pseudo_label=update_pseudo_label,
                                                                            threshold=threshold)
        ensemble_result_distribution = calculate_ensemble_label(dict_dict=copy.deepcopy(model_hub))
        ensemble_result_distribution = doing_few_shot_change(arr=copy.deepcopy(ensemble_result_distribution), labels=labels, index_list=labeling_data_index_list)
        return ensemble_result_distribution



def pesudo_gen_all(labels, Model_result_dict, labeling_budget_ratio, Uncertainty_sampling_strategy='Entropy'):
    _, first_value = list(Model_result_dict.items())[0]
    all_data_number, _ = first_value.shape
    labeling_budget_max_number = int(labeling_budget_ratio * all_data_number)
    batch_size_labeling_num = int(labeling_budget_max_number / 10)
    labeling_data_index_list = []
    model_hub = copy.deepcopy(Model_result_dict)
    pseudo_label = calculate_ensemble_label(copy.deepcopy(model_hub))
    model_hub = copy.deepcopy(Model_result_dict)
    update_pseudo_label, Gt_index = Active_Label_Acquisition(ensemble_result_distribution=copy.deepcopy(pseudo_label),
                                                           update_budget=batch_size_labeling_num,
                                                           labeling_data_index_list=labeling_data_index_list,
                                                           Uncertainty_sampling_strategy=Uncertainty_sampling_strategy,
                                                           labeling_budget_max_number=labeling_budget_max_number,
                                                           labels=labels)

    merged_array = np.concatenate((np.array(labeling_data_index_list), copy.deepcopy(Gt_index))).astype(int)
    labeling_data_index_list, _ = np.unique(merged_array, return_counts=True)
    labeling_data_index_list = labeling_data_index_list.tolist()
    ensemble_result_distribution = calculate_ensemble_label(dict_dict=copy.deepcopy(model_hub))
    ensemble_result_distribution = doing_few_shot_change(arr=copy.deepcopy(ensemble_result_distribution), labels=labels, index_list=labeling_data_index_list)
    return ensemble_result_distribution







def Random_Few_shot_change(ensemble_result_distribution, ratio,labels):
    if ratio == 0:
        return ensemble_result_distribution
    _, val_data_number, _ = ensemble_result_distribution.shape
    N = int(val_data_number * ratio)
    random_numbers = np.array(random.sample(range(val_data_number), N))
    ensemble_result_distribution = doing_few_shot_change(copy.deepcopy(ensemble_result_distribution), labels, random_numbers)
    return ensemble_result_distribution






def Few_shot_change(ensemble_result_distribution, ratio,labels):
    if ratio == 0:
        return ensemble_result_distribution
    Few_shot_Soft_ensemble_result_distribution = normalized_entropy(copy.deepcopy(ensemble_result_distribution))
    _, val_data_number, _ = ensemble_result_distribution.shape
    N = int(val_data_number * ratio)
    Gt_index = max_k_indices(Few_shot_Soft_ensemble_result_distribution, N)
    ensemble_result_distribution = doing_few_shot_change(copy.deepcopy(ensemble_result_distribution), labels, Gt_index)
    return ensemble_result_distribution




def create_zeros_matrix_same_shape(matrix: np.ndarray) -> np.ndarray:
    return np.zeros(matrix.shape)







def get_argmax(matrix: np.ndarray) -> np.ndarray:
    return np.argmax(matrix, axis=1)

def compute_acc(arr1, arr2):
    arr2 = np.squeeze(arr2)
    arr1 = get_argmax(arr1)
    arr2 = get_argmax(arr2)
    if len(arr1) != len(arr2):
        raise ValueError("Both arrays should have the same length.")
    equal_elements = np.sum(arr1 == arr2)
    ratio_rate = equal_elements / len(arr1)
    return ratio_rate



def dict_values_to_list(d):
    return list(d.values())


def sort_dict_by_value_insc(my_dict):
    sorted_dict = sorted(my_dict.items(), key=lambda x: x[1])
    sorted_keys = [item[0] for item in sorted_dict]
    return sorted_keys


def sort_dict_by_value_desc(my_dict):
    sorted_dict = sorted(my_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_keys = [item[0] for item in sorted_dict]
    return sorted_keys




def shuffled_list(N):
    lst = list(range(1, N + 1))
    random.shuffle(lst)
    return lst




def plot_ratio_lines(ax, x, y_lists, titles=None, dataset_name="Multiple Lines Plot",type = 'acc'):
    AXIS_LABELSIZE = 23
    TICK_LABELSIZE = 25
    lines = []
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H', '+', 'x', '|', '_']
    custom_colors = [
        'red', 'green', 'purple', 'orange', 'blue',
        'pink', 'lime', 'violet', 'brown', 'navy',
        'magenta', 'cyan', 'crimson', 'darkgreen', 'indigo',
        'salmon', 'lightgreen', 'plum', 'coral', 'darkblue'
    ]
    if len(y_lists) > len(custom_colors):
        custom_colors *= (len(y_lists) // len(custom_colors)) + 1
    for i, (y, color) in enumerate(zip(y_lists, custom_colors)):
        if "Random" in titles[i]:
            line, = ax.plot(x, y, linestyle='--', marker=markers[i % len(markers)], color=color)
        else:
            line, = ax.plot(x, y, color=color)
        lines.append(line)

    ax.set_xlabel("ratio", fontsize=AXIS_LABELSIZE)
    if type == 'acc':
        ax.set_ylabel("Ranking Correction", fontsize=AXIS_LABELSIZE)
    else:
        ax.set_ylabel("Optimal Gap", fontsize=AXIS_LABELSIZE)
    ax.set_title(dataset_name, fontsize=AXIS_LABELSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
    ax.grid(True)
    return lines


