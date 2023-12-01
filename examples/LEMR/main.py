import tqdm
import copy
import numpy as np
import os
import pickle
import argparse
from functions import  gen_ensemble_matrix,gen_ensemble_result_from_matrix,\
    pesudo_gen, pesudo_gen_all, Random_Few_shot_change,Few_shot_change,\
    compute_acc,create_zeros_matrix_same_shape,\
    sort_dict_by_value_desc,sort_list_by_dict,string_list_correlation
from morabench.utils import setup_seed,accuracy,read_pkl_file
from morabench.conifg import root_path

ratio_list = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
              0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, ]
Ensemble_method_list = ['hard', 'soft']
Acquisition_list = ['Entropy', 'Uncertainty', 'Margin', 'False', ]
threshold = -2.8
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int,default=0, help="seed")
parser.add_argument("--Ensemble_method_id", type=int,default=1, help="Ensemble_method_id")
parser.add_argument("--total_split_number", type=int,default=50, help="total_split_number")
parser.add_argument("--dataset_name", type=str,default='story', help="dataset_name")
parser.add_argument("--model_committee_type", type=str,default='z_score', help="model_committee_type")
args = parser.parse_args()
dataset_name = args.dataset_name
setup_seed(args.seed)
result_save_path = f'{root_path}/Extracting_information_data/{dataset_name}'
dataset_matrix_set = read_pkl_file(f'{result_save_path}/result.pkl')
dataset_matrix_set[dataset_name]['labels'] = dataset_matrix_set[dataset_name]['labels'] .cpu().numpy()
model_to_idx = {}
total_split_number = args.total_split_number
Ensemble_method = Ensemble_method_list[args.Ensemble_method_id]
img_data_save_dict = {}
img_data_save_dict['rc'] = {}
img_data_save_dict['og'] = {}
optimal_gap_value_dict = {}
for acq_method in Acquisition_list:
    optimal_gap_value_dict[f'{acq_method}'] = np.array([])
ranking_correction_value_dict = {}
for acq_method in Acquisition_list:
    ranking_correction_value_dict[f'{acq_method}'] = np.array([])
for split_num in tqdm.tqdm(range(total_split_number)):
    val_indices_list = read_pkl_file(f'{root_path}/Extracting_information_data/{dataset_name}/val_split.pkl')
    test_indices_list = read_pkl_file(f'{root_path}/Extracting_information_data/{dataset_name}/test_split.pkl')
    val_indices, test_indices = val_indices_list[dataset_name][split_num], test_indices_list[dataset_name][split_num]
    Soft_ensemble_dict = copy.deepcopy(dataset_matrix_set[dataset_name]['Soft_ensemble_dict'])
    Hard_ensemble_dict = copy.deepcopy(dataset_matrix_set[dataset_name]['Hard_ensemble_dict'])
    Soft_ensemble_matrix = gen_ensemble_matrix(Soft_ensemble_dict)
    Hard_ensemble_matrix = gen_ensemble_matrix(Hard_ensemble_dict)
    Soft_ensemble_result, Soft_ensemble_result_distribution = gen_ensemble_result_from_matrix(Soft_ensemble_matrix)
    Hard_ensemble_result, Hard_ensemble_result_distribution = gen_ensemble_result_from_matrix(Hard_ensemble_matrix)
    if Ensemble_method == 'hard':
        Model_result_dict = copy.deepcopy(Hard_ensemble_dict)
        result_distribution = copy.deepcopy(Hard_ensemble_result_distribution)
    else:
        Model_result_dict = copy.deepcopy(Soft_ensemble_dict)
        result_distribution = copy.deepcopy(Soft_ensemble_result_distribution)
    for rk in Model_result_dict:
        Model_result_dict[rk] = Model_result_dict[rk][val_indices, :]
    labels = copy.deepcopy(dataset_matrix_set[dataset_name]['labels'][val_indices])
    result_dict_for_plot = {}
    for acq_method in Acquisition_list:
        result_dict_for_plot[f'{acq_method}'] = {}
    for Few_shot_if in Acquisition_list:
        for ratio in ratio_list:
            ensemble_result_distribution = copy.deepcopy(result_distribution[:, val_indices, :])
            result_dict_for_plot[Few_shot_if][ratio] = {}
            if Few_shot_if == 'False':
                ensemble_result_distribution = Random_Few_shot_change(copy.deepcopy(ensemble_result_distribution), ratio=ratio,labels=labels)
            elif Few_shot_if == 'True':
                ensemble_result_distribution = Few_shot_change(copy.deepcopy(ensemble_result_distribution), ratio=ratio,labels=labels)
            else:
                if ratio == 0:
                    ensemble_result_distribution = copy.deepcopy(ensemble_result_distribution)
                else:
                    if args.model_committee_type == 'z_score':
                        ensemble_result_distribution = pesudo_gen(labels=copy.deepcopy(labels),
                                                                          Model_result_dict=copy.deepcopy(Model_result_dict),
                                                                          labeling_budget_ratio=copy.deepcopy(ratio),
                                                                          Uncertainty_sampling_strategy = Few_shot_if,
                                                                          threshold=threshold,
                                                                          )
                    else:
                        ensemble_result_distribution = pesudo_gen_all(labels=copy.deepcopy(labels),
                                                                          Model_result_dict=copy.deepcopy(Model_result_dict),
                                                                          labeling_budget_ratio=copy.deepcopy(ratio),
                                                                          Uncertainty_sampling_strategy = Few_shot_if,
                                                                          )
            ground_truth_distribution = copy.deepcopy(ensemble_result_distribution)
            model_ranking_tmp_value_save_softsensmble = {}
            for model_name in Model_result_dict:
                model_output_distribution = Model_result_dict[model_name]
                metric_value = compute_acc(copy.deepcopy(model_output_distribution), copy.deepcopy(ground_truth_distribution))
                model_ranking_tmp_value_save_softsensmble[model_name] = metric_value
            ground_truth_distribution = create_zeros_matrix_same_shape(ensemble_result_distribution)
            _, num_data, _ = ground_truth_distribution.shape
            for k in range(num_data):
                ground_truth_distribution[0][k][labels[k]] = 1
            model_ranking_tmp_value_save_labels = {}
            for model_name in Model_result_dict:
                model_output_distribution = Model_result_dict[model_name]
                metric_value = compute_acc(copy.deepcopy(model_output_distribution), copy.deepcopy(ground_truth_distribution))
                model_ranking_tmp_value_save_labels[model_name] = metric_value
            model_to_idx[dataset_name] = {}
            for key_idx, key in enumerate(dataset_matrix_set[dataset_name]['Hard_ensemble_dict']):
                model_to_idx[dataset_name][key] = key_idx
            val_ranking_key_list_softsensmble = sort_dict_by_value_desc(copy.deepcopy(model_ranking_tmp_value_save_softsensmble))
            val_ranking_key_list_label = sort_dict_by_value_desc(copy.deepcopy(model_ranking_tmp_value_save_labels))
            best_model_name_ensemble = val_ranking_key_list_softsensmble[0]
            best_model_name_label = val_ranking_key_list_label[0]
            if Ensemble_method == 'hard':
                test_Model_result_dict = copy.deepcopy(Hard_ensemble_dict)
            else:
                test_Model_result_dict = copy.deepcopy(Soft_ensemble_dict)
            test_select_model_output_ensemble = test_Model_result_dict[best_model_name_ensemble][test_indices]
            test_select_model_output_label = test_Model_result_dict[best_model_name_label][test_indices]
            test_labels = copy.deepcopy(dataset_matrix_set[dataset_name]['labels'][test_indices])
            # rc +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            val_Acc_ranking_idx_list_softsensmble = sort_list_by_dict(val_ranking_key_list_softsensmble, model_to_idx[dataset_name])
            val_Acc_ranking_idx_list_labels = sort_list_by_dict(val_ranking_key_list_label, model_to_idx[dataset_name])
            rank_correlation, _ = string_list_correlation(list1=val_ranking_key_list_label, list2=val_ranking_key_list_softsensmble)
            result_dict_for_plot[Few_shot_if][ratio]['rank_correlation'] = rank_correlation

            # og +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            acc_ensemble_result = accuracy(np.array([test_select_model_output_ensemble]),test_labels)
            acc_label_result = accuracy(np.array([test_select_model_output_label]), test_labels)
            optimal_gap = acc_ensemble_result - acc_label_result
            result_dict_for_plot[Few_shot_if][ratio]['optimal_gap'] = optimal_gap

    acq_method_rc_list = {}
    for acq_method in Acquisition_list:
        acq_method_rc_list[f'{acq_method}']=[]
    for ratio in ratio_list:
        for acq_method in Acquisition_list:
            acq_method_rc_list[f'{acq_method}'].append(result_dict_for_plot[f'{acq_method}'][ratio]['rank_correlation'])
    for acq_method in Acquisition_list:
        if len(ranking_correction_value_dict[f'{acq_method}']) == 0:
            ranking_correction_value_dict[f'{acq_method}'] =np.array(acq_method_rc_list[f'{acq_method}'])
        else:
            ranking_correction_value_dict[f'{acq_method}'] += np.array(acq_method_rc_list[f'{acq_method}'])
    acq_method_og_list = {}
    for acq_method in Acquisition_list:
        acq_method_og_list[f'{acq_method}']=[]
    for ratio in ratio_list:
        for acq_method in Acquisition_list:
            acq_method_og_list[f'{acq_method}'].append(result_dict_for_plot[f'{acq_method}'][ratio]['optimal_gap'])
    for acq_method in Acquisition_list:
        if len(optimal_gap_value_dict[f'{acq_method}']) == 0:
            optimal_gap_value_dict[f'{acq_method}'] =np.array(acq_method_og_list[f'{acq_method}'])
        else:
            optimal_gap_value_dict[f'{acq_method}'] += np.array(acq_method_og_list[f'{acq_method}'])
rc_metric_list = []
for acq_method in Acquisition_list:
    rc_metric_list.append(ranking_correction_value_dict[f'{acq_method}']/ total_split_number)
img_data_save_dict['rc'][dataset_name] = {
    'ratio_list':ratio_list,
    'all_metric_list':rc_metric_list,
    'name_metric_list':Acquisition_list,
    'sub_name':dataset_name,
}
og_metric_list = []
for acq_method in Acquisition_list:
    og_metric_list.append(optimal_gap_value_dict[f'{acq_method}']/ total_split_number)
img_data_save_dict['og'][dataset_name] = {
    'ratio_list':ratio_list,
    'all_metric_list':og_metric_list,
    'name_metric_list':Acquisition_list,
    'sub_name':dataset_name,
}
if args.model_committee_type == 'z_score':
    img_data_save_path = f'{root_path}/Extracting_information_data/dataset_name_{dataset_name}/' \
                         f'Ensemble_method_{Ensemble_method}/seed_{args.seed}/'
else:
    img_data_save_path = f'{root_path}/Extracting_information_data/all_model/dataset_name_{dataset_name}/' \
                         f'Ensemble_method_{Ensemble_method}/seed_{args.seed}/'
print(img_data_save_path)
if not os.path.exists(img_data_save_path):
    os.makedirs(img_data_save_path)

with open(f'{img_data_save_path}draw_image_data.pkl', 'wb') as f:
    pickle.dump(img_data_save_dict, f)