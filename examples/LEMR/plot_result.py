import matplotlib.gridspec as gridspec
import copy
import matplotlib.pyplot as plt
import argparse
from functions import plot_ratio_lines
from morabench.conifg import root_path, ensemble_map,unc_method_map,\
    usb_dataset_dict_map,usb_dataset_dict
from morabench.utils import read_pkl_file



ratio_list = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
              0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, ]
Ensemble_method_list = ['hard', 'soft']
dataset_dict_map = usb_dataset_dict_map
dataset_dict = usb_dataset_dict
parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str,default='rc', help="rc or og")
args = parser.parse_args()
metric = args.metric
plot_dict = {}
for idx, sub_name in enumerate(dataset_dict):
    plot_dict[sub_name]={}
    plot_dict[sub_name]['rc']={}
    plot_dict[sub_name]['og']={}
for Ensemble_method in Ensemble_method_list:
    for idx, sub_name in enumerate(dataset_dict):
        img_data_save_path = f'{root_path}/Extracting_information_data/all_model/dataset_name_{sub_name}/' \
                             f'Ensemble_method_{Ensemble_method}/seed_{args.seed}/'
        data_path = img_data_save_path+ 'draw_image_data.pkl'
        data = read_pkl_file(data_path)
        if not 'all_metric_list' in plot_dict[sub_name]['rc']:
            plot_dict[sub_name]['rc']['ratio_list'] = data['rc'][sub_name]['ratio_list']
            plot_dict[sub_name]['og']['ratio_list'] = data['og'][sub_name]['ratio_list']
            plot_dict[sub_name]['rc']['name_metric_list'] = []
            for unc_method in data['rc'][sub_name]['name_metric_list']:
                plot_dict[sub_name]['rc']['name_metric_list'].append(ensemble_map[Ensemble_method]+'+'
                                                                      +unc_method_map[unc_method]+'+'
                                                                        'All Model'
                                                                      )
            plot_dict[sub_name]['og']['name_metric_list'] = copy.deepcopy(plot_dict[sub_name]['rc']['name_metric_list'])
            plot_dict[sub_name]['rc']['all_metric_list'] = data['rc'][sub_name]['all_metric_list']
            plot_dict[sub_name]['og']['all_metric_list'] = data['og'][sub_name]['all_metric_list']
        else:
            for unc_method in data['rc'][sub_name]['name_metric_list']:
                plot_dict[sub_name]['rc']['name_metric_list'].append(ensemble_map[Ensemble_method]+'+'
                                                                      +unc_method_map[unc_method]+'+'
                                                                        'All Model'
                                                                      )
            for unc_method in data['og'][sub_name]['name_metric_list']:
                plot_dict[sub_name]['og']['name_metric_list'].append(ensemble_map[Ensemble_method]+'+'
                                                                      +unc_method_map[unc_method]+'+'
                                                                        'All Model'
                                                                      )
            plot_dict[sub_name]['rc']['all_metric_list']  += data['rc'][sub_name]['all_metric_list']
            plot_dict[sub_name]['og']['all_metric_list']  += data['og'][sub_name]['all_metric_list']


plot_dict_tmp = {}
for idx, sub_name in enumerate(dataset_dict):
    plot_dict_tmp[sub_name]={}
    plot_dict_tmp[sub_name]['rc']={}
    plot_dict_tmp[sub_name]['og']={}
for Ensemble_method in Ensemble_method_list:
    for idx, sub_name in enumerate(dataset_dict):
        img_data_save_path = f'{root_path}/Extracting_information_data/dataset_name_{sub_name}/' \
                             f'Ensemble_method_{Ensemble_method}/seed_{args.seed}/'
        data_path = img_data_save_path+ 'draw_image_data.pkl'
        data = read_pkl_file(data_path)
        if not 'all_metric_list' in plot_dict_tmp[sub_name]['rc']:
            plot_dict_tmp[sub_name]['rc']['ratio_list'] = data['rc'][sub_name]['ratio_list']
            plot_dict_tmp[sub_name]['og']['ratio_list'] = data['og'][sub_name]['ratio_list']
            plot_dict_tmp[sub_name]['rc']['name_metric_list'] = []
            for unc_method in data['rc'][sub_name]['name_metric_list']:
                plot_dict_tmp[sub_name]['rc']['name_metric_list'].append(ensemble_map[Ensemble_method]+'+'
                                                                      +unc_method_map[unc_method]+'+'
                                                                        'Z-score'
                                                                      )
            plot_dict_tmp[sub_name]['og']['name_metric_list'] = copy.deepcopy(plot_dict_tmp[sub_name]['rc']['name_metric_list'])
            plot_dict_tmp[sub_name]['rc']['all_metric_list'] = data['rc'][sub_name]['all_metric_list']
            plot_dict_tmp[sub_name]['og']['all_metric_list'] = data['og'][sub_name]['all_metric_list']

        else:

            for unc_method in data['rc'][sub_name]['name_metric_list']:
                plot_dict_tmp[sub_name]['rc']['name_metric_list'].append(ensemble_map[Ensemble_method]+'+'
                                                                      +unc_method_map[unc_method]+'+'
                                                                        'Z-score'
                                                                      )
            for unc_method in data['og'][sub_name]['name_metric_list']:
                plot_dict_tmp[sub_name]['og']['name_metric_list'].append(ensemble_map[Ensemble_method]+'+'
                                                                      +unc_method_map[unc_method]+'+'
                                                                        'Z-score'
                                                                      )
            plot_dict_tmp[sub_name]['rc']['all_metric_list']  += data['rc'][sub_name]['all_metric_list']
            plot_dict_tmp[sub_name]['og']['all_metric_list']  += data['og'][sub_name]['all_metric_list']

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
                              y_lists= y_lists,
                              titles=titles,
                              dataset_name=dataset_dict_map[sub_name],
                              type = metric,
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
