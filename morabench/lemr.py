import tqdm
import numpy as np
import os
from morabench.utils import setup_seed,read_pkl_file,\
    pesudo_gen, pesudo_gen_all, Random_Few_shot_change,\
    compute_acc,create_zeros_matrix_same_shape,\
    sort_dict_by_value_desc,plot_ratio_lines
from morabench.evaluation import ranking_correction_comp, optimal_gap_comp
from morabench.conifg import *
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import copy
import pickle
from typing import Any, Optional
from morabench import rank_base_method
from morabench.model_set import Model_set

class LEMR(rank_base_method):
    def __init__(self,
                 model_set_path:Optional[str] = '',
                 seed: Optional[int] = 0,
                 Ensemble_method: Optional[str] = '',
                 total_split_number: Optional[int] = 50,
                 dataset_name: Optional[str] = '',
                 model_committee_type: Optional[str] = '',
                 result_save_path: Optional[str] = '',
                 **kwargs: Any):

        super().__init__()
        self.input_hyperparas = {
            'model_set_path': model_set_path,
            'seed': seed,
            'Ensemble_method': Ensemble_method,
            'total_split_number': total_split_number,
            'dataset_name': dataset_name,
            'model_committee_type': model_committee_type,
            'result_save_path':result_save_path
        }
        self.model_set = Model_set()
        self.result_data_dict={}
    def load(self) -> None:
        self.model_set.load_model_set(load_path=self.input_hyperparas['model_set_path'])
    def save(self) -> None:
        """Result save.
        ----------
        """
        total_split_number = self.input_hyperparas['total_split_number']
        dataset_name = self.input_hyperparas['dataset_name']
        result_save_path = self.input_hyperparas['result_save_path']
        rc_dict = self.result_data_dict['ranking_correction']
        og_dict = self.result_data_dict['optimal_gap']

        result_data = {}
        result_data['rc'] = {}
        result_data['og'] = {}

        rc_metric_list = []
        for acq_method in Acquisition_list:
            rc_metric_list.append(rc_dict[f'{acq_method}'] / total_split_number)
        result_data['rc'][dataset_name] = {
            'ratio_list': ratio_list,
            'all_metric_list': rc_metric_list,
            'name_metric_list': Acquisition_list,
            'sub_name': dataset_name,
        }
        og_metric_list = []
        for acq_method in Acquisition_list:
            og_metric_list.append(og_dict[f'{acq_method}'] / total_split_number)
        result_data['og'][dataset_name] = {
            'ratio_list': ratio_list,
            'all_metric_list': og_metric_list,
            'name_metric_list': Acquisition_list,
            'sub_name': dataset_name,
        }
        with open(f'{result_save_path}draw_image_data.pkl', 'wb') as f:
            pickle.dump(result_data, f)

    def metric_value_compute(self) -> None:
        seed = self.input_hyperparas['seed']
        Ensemble_method = self.input_hyperparas['Ensemble_method']
        total_split_number = self.input_hyperparas['total_split_number']
        dataset_name = self.input_hyperparas['dataset_name']
        model_committee_type = self.input_hyperparas['model_committee_type']
        result_save_path = self.input_hyperparas['result_save_path']
        model_set = self.model_set

        if not os.path.exists(result_save_path):
            print(f"data not exist, start running.")
        else:
            print(f"data all ready exist.")
            return
        setup_seed(seed)
        model_set.dataset_matrix_set[dataset_name]['labels'] = model_set.dataset_matrix_set[dataset_name]['labels'].cpu().numpy()
        model_to_idx = {}
        og_dict = {}
        for acq_method in Acquisition_list:
            og_dict[f'{acq_method}'] = np.array([])
        rc_dict = {}
        for acq_method in Acquisition_list:
            rc_dict[f'{acq_method}'] = np.array([])
        val_split = read_pkl_file(f'{root_path}/Extracting_information_data/{dataset_name}/val_split.pkl')
        test_split = read_pkl_file(f'{root_path}/Extracting_information_data/{dataset_name}/test_split.pkl')
        for split_num in tqdm.tqdm(range(total_split_number)):
            val_indices, test_indices = val_split[dataset_name][split_num], test_split[dataset_name][split_num]
            result_distribution = copy.deepcopy(model_set.result_distribution[dataset_name])
            for rk in model_set.Model_result_dict[dataset_name]:
                model_set.Model_result_dict[dataset_name][rk] = model_set.Model_result_dict[dataset_name][rk][val_indices, :]
            labels = copy.deepcopy(model_set.dataset_matrix_set[dataset_name]['labels'][val_indices])
            Acquisition_result_dict = {}
            for acq_method in Acquisition_list:
                Acquisition_result_dict[f'{acq_method}'] = {}
            for acq_method in Acquisition_list:
                for ratio in ratio_list:
                    ensemble_result_distribution = copy.deepcopy(result_distribution[:, val_indices, :])
                    Acquisition_result_dict[acq_method][ratio] = {}
                    if acq_method == 'False':
                        ensemble_result_distribution = Random_Few_shot_change(copy.deepcopy(ensemble_result_distribution), ratio=ratio, labels=labels)
                    else:
                        if ratio == 0:
                            ensemble_result_distribution = copy.deepcopy(ensemble_result_distribution)
                        else:
                            if model_committee_type == 'z_score':
                                ensemble_result_distribution = pesudo_gen(labels=copy.deepcopy(labels),
                                                                          Model_result_dict=copy.deepcopy(model_set.Model_result_dict[dataset_name]),
                                                                          labeling_budget_ratio=copy.deepcopy(ratio),
                                                                          Uncertainty_sampling_strategy=acq_method,
                                                                          threshold=threshold,
                                                                          )
                            else:
                                ensemble_result_distribution = pesudo_gen_all(labels=copy.deepcopy(labels),
                                                                              Model_result_dict=copy.deepcopy(model_set.Model_result_dict[dataset_name]),
                                                                              labeling_budget_ratio=copy.deepcopy(ratio),
                                                                              Uncertainty_sampling_strategy=acq_method,
                                                                              )
                    ground_truth_distribution = copy.deepcopy(ensemble_result_distribution)
                    model_ranking_tmp_value_save_softsensmble = {}
                    for model_name in model_set.Model_result_dict[dataset_name]:
                        model_output_distribution = model_set.Model_result_dict[dataset_name][model_name]
                        metric_value = compute_acc(copy.deepcopy(model_output_distribution), copy.deepcopy(ground_truth_distribution))
                        model_ranking_tmp_value_save_softsensmble[model_name] = metric_value
                    ground_truth_distribution = create_zeros_matrix_same_shape(ensemble_result_distribution)
                    _, num_data, _ = ground_truth_distribution.shape
                    for k in range(num_data):
                        ground_truth_distribution[0][k][labels[k]] = 1
                    model_ranking_tmp_value_save_labels = {}
                    for model_name in model_set.Model_result_dict[dataset_name]:
                        model_output_distribution = model_set.Model_result_dict[dataset_name][model_name]
                        metric_value = compute_acc(copy.deepcopy(model_output_distribution), copy.deepcopy(ground_truth_distribution))
                        model_ranking_tmp_value_save_labels[model_name] = metric_value
                    model_to_idx[dataset_name] = {}
                    for key_idx, key in enumerate(model_set.dataset_matrix_set[dataset_name]['Hard_ensemble_dict']):
                        model_to_idx[dataset_name][key] = key_idx
                    pseudo_label_rank = sort_dict_by_value_desc(copy.deepcopy(model_ranking_tmp_value_save_softsensmble))
                    full_label_rank = sort_dict_by_value_desc(copy.deepcopy(model_ranking_tmp_value_save_labels))
                    top_model_selected_by_pseudo_label = pseudo_label_rank[0]
                    top_model_selected_by_full_label = full_label_rank[0]
                    if Ensemble_method == 'hard':
                        test_Model_result_dict = copy.deepcopy(model_set.Model_result_dict[dataset_name][dataset_name]['Hard_ensemble_dict'])
                    else:
                        test_Model_result_dict = copy.deepcopy(model_set.Model_result_dict[dataset_name][dataset_name]['Soft_ensemble_dict'])
                    pseudo_label_test_selection = test_Model_result_dict[top_model_selected_by_pseudo_label][test_indices]
                    full_label_test_selection = test_Model_result_dict[top_model_selected_by_full_label][test_indices]
                    test_labels = copy.deepcopy(model_set.dataset_matrix_set[dataset_name]['labels'][test_indices])
                    # rc +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    rank_correlation = ranking_correction_comp(list1=full_label_rank, list2=pseudo_label_rank)
                    Acquisition_result_dict[acq_method][ratio]['rank_correlation'] = rank_correlation

                    # og +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    optimal_gap = optimal_gap_comp(list1=pseudo_label_test_selection, list2=full_label_test_selection, test_labels=test_labels)
                    Acquisition_result_dict[acq_method][ratio]['optimal_gap'] = optimal_gap



            acq_method_rc_list = {}
            for acq_method in Acquisition_list:
                acq_method_rc_list[f'{acq_method}'] = []
            for ratio in ratio_list:
                for acq_method in Acquisition_list:
                    acq_method_rc_list[f'{acq_method}'].append(Acquisition_result_dict[f'{acq_method}'][ratio]['rank_correlation'])
            for acq_method in Acquisition_list:
                if len(rc_dict[f'{acq_method}']) == 0:
                    rc_dict[f'{acq_method}'] = np.array(acq_method_rc_list[f'{acq_method}'])
                else:
                    rc_dict[f'{acq_method}'] += np.array(acq_method_rc_list[f'{acq_method}'])
            acq_method_og_list = {}
            for acq_method in Acquisition_list:
                acq_method_og_list[f'{acq_method}'] = []
            for ratio in ratio_list:
                for acq_method in Acquisition_list:
                    acq_method_og_list[f'{acq_method}'].append(Acquisition_result_dict[f'{acq_method}'][ratio]['optimal_gap'])
            for acq_method in Acquisition_list:
                if len(og_dict[f'{acq_method}']) == 0:
                    og_dict[f'{acq_method}'] = np.array(acq_method_og_list[f'{acq_method}'])
                else:
                    og_dict[f'{acq_method}'] += np.array(acq_method_og_list[f'{acq_method}'])

        self.result_data_dict['optimal_gap']=og_dict
        self.result_data_dict['ranking_correction'] = rc_dict

    def rank(self, *args: Any, **kwargs: Any) -> None:
        self.load()
        self.metric_value_compute()
        self.save()

def lemr_show(metric_type,seed) -> None:
    Ensemble_method_list = ['hard', 'soft']
    dataset_dict_map = usb_dataset_dict_map
    dataset_dict = usb_dataset_dict
    metric = metric_type
    plot_dict = {}
    for idx, sub_name in enumerate(dataset_dict):
        plot_dict[sub_name] = {}
        plot_dict[sub_name]['rc'] = {}
        plot_dict[sub_name]['og'] = {}
    for Ensemble_method in Ensemble_method_list:
        for idx, sub_name in enumerate(dataset_dict):
            result_save_path = f'{root_path}/Extracting_information_data/all_model/dataset_name_{sub_name}/' \
                                 f'Ensemble_method_{Ensemble_method}/seed_{seed}/'
            data_path = result_save_path + 'draw_image_data.pkl'
            data = read_pkl_file(data_path)
            if not 'all_metric_list' in plot_dict[sub_name]['rc']:
                plot_dict[sub_name]['rc']['ratio_list'] = data['rc'][sub_name]['ratio_list']
                plot_dict[sub_name]['og']['ratio_list'] = data['og'][sub_name]['ratio_list']
                plot_dict[sub_name]['rc']['name_metric_list'] = []
                for unc_method in data['rc'][sub_name]['name_metric_list']:
                    plot_dict[sub_name]['rc']['name_metric_list'].append(ensemble_map[Ensemble_method] + '+'
                                                                         + unc_method_map[unc_method] + '+'
                                                                                                        'All Model'
                                                                         )
                plot_dict[sub_name]['og']['name_metric_list'] = copy.deepcopy(plot_dict[sub_name]['rc']['name_metric_list'])
                plot_dict[sub_name]['rc']['all_metric_list'] = data['rc'][sub_name]['all_metric_list']
                plot_dict[sub_name]['og']['all_metric_list'] = data['og'][sub_name]['all_metric_list']
            else:
                for unc_method in data['rc'][sub_name]['name_metric_list']:
                    plot_dict[sub_name]['rc']['name_metric_list'].append(ensemble_map[Ensemble_method] + '+'
                                                                         + unc_method_map[unc_method] + '+'
                                                                                                        'All Model'
                                                                         )
                for unc_method in data['og'][sub_name]['name_metric_list']:
                    plot_dict[sub_name]['og']['name_metric_list'].append(ensemble_map[Ensemble_method] + '+'
                                                                         + unc_method_map[unc_method] + '+'
                                                                                                        'All Model'
                                                                         )
                plot_dict[sub_name]['rc']['all_metric_list'] += data['rc'][sub_name]['all_metric_list']
                plot_dict[sub_name]['og']['all_metric_list'] += data['og'][sub_name]['all_metric_list']

    plot_dict_tmp = {}
    for idx, sub_name in enumerate(dataset_dict):
        plot_dict_tmp[sub_name] = {}
        plot_dict_tmp[sub_name]['rc'] = {}
        plot_dict_tmp[sub_name]['og'] = {}
    for Ensemble_method in Ensemble_method_list:
        for idx, sub_name in enumerate(dataset_dict):
            result_save_path = f'{root_path}/Extracting_information_data/dataset_name_{sub_name}/' \
                                 f'Ensemble_method_{Ensemble_method}/seed_{seed}/'
            data_path = result_save_path + 'draw_image_data.pkl'
            data = read_pkl_file(data_path)
            if not 'all_metric_list' in plot_dict_tmp[sub_name]['rc']:
                plot_dict_tmp[sub_name]['rc']['ratio_list'] = data['rc'][sub_name]['ratio_list']
                plot_dict_tmp[sub_name]['og']['ratio_list'] = data['og'][sub_name]['ratio_list']
                plot_dict_tmp[sub_name]['rc']['name_metric_list'] = []
                for unc_method in data['rc'][sub_name]['name_metric_list']:
                    plot_dict_tmp[sub_name]['rc']['name_metric_list'].append(ensemble_map[Ensemble_method] + '+'
                                                                             + unc_method_map[unc_method] + '+'
                                                                                                            'Z-score'
                                                                             )
                plot_dict_tmp[sub_name]['og']['name_metric_list'] = copy.deepcopy(plot_dict_tmp[sub_name]['rc']['name_metric_list'])
                plot_dict_tmp[sub_name]['rc']['all_metric_list'] = data['rc'][sub_name]['all_metric_list']
                plot_dict_tmp[sub_name]['og']['all_metric_list'] = data['og'][sub_name]['all_metric_list']

            else:

                for unc_method in data['rc'][sub_name]['name_metric_list']:
                    plot_dict_tmp[sub_name]['rc']['name_metric_list'].append(ensemble_map[Ensemble_method] + '+'
                                                                             + unc_method_map[unc_method] + '+'
                                                                                                            'Z-score'
                                                                             )
                for unc_method in data['og'][sub_name]['name_metric_list']:
                    plot_dict_tmp[sub_name]['og']['name_metric_list'].append(ensemble_map[Ensemble_method] + '+'
                                                                             + unc_method_map[unc_method] + '+'
                                                                                                            'Z-score'
                                                                             )
                plot_dict_tmp[sub_name]['rc']['all_metric_list'] += data['rc'][sub_name]['all_metric_list']
                plot_dict_tmp[sub_name]['og']['all_metric_list'] += data['og'][sub_name]['all_metric_list']
    fig = plt.figure(figsize=(40, 10))
    gs = gridspec.GridSpec(2, 5, figure=fig)
    axes_row1 = [fig.add_subplot(gs[0, i]) for i in range(5)]
    axes_row2 = [fig.add_subplot(gs[1, i]) for i in range(5)]
    axes = axes_row1 + axes_row2
    for idx, (ax, sub_name) in enumerate(zip(axes, dataset_dict)):
        y_lists = plot_dict[sub_name][metric]['all_metric_list'] + plot_dict_tmp[sub_name][metric]['all_metric_list']
        titles = plot_dict[sub_name][metric]['name_metric_list'] + plot_dict_tmp[sub_name][metric]['name_metric_list']
        x = plot_dict[sub_name][metric]['ratio_list']
        new_lines = plot_ratio_lines(ax=ax,
                                     x=x,
                                     y_lists=y_lists,
                                     titles=titles,
                                     dataset_name=dataset_dict_map[sub_name],
                                     type=metric,
                                     )
        if idx == 0:
            lines = new_lines
            titles = plot_dict[sub_name][metric]['name_metric_list'] + plot_dict_tmp[sub_name][metric]['name_metric_list']
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    legend_ax = fig.add_axes([0.1, 0.92, 0.8, 0.05], frame_on=False)
    legend_ax.axis('off')
    legend = fig.legend(lines, titles, loc='upper center', bbox_to_anchor=(0.513, 1.2),
                        ncol=4, prop={'size': 25}, handlelength=3)
    for line in legend.get_lines():
        line.set_linewidth(4.0)
    plt.show()