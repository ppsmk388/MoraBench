import pickle
import numpy as np
import os
import argparse
from conifg import root_path
from utils import setup_seed
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str,default='story', help="dataset_name")
parser.add_argument("--seed", type=int,default=42, help="seed")
parser.add_argument("--split_num", type=int,default=50, help="split_num")
args = parser.parse_args()
setup_seed(args.seed)
dataset_name = args.dataset_name
split_num = args.split_num
result_save_path = f'{root_path}/Extracting_information_data/{dataset_name}'
with open(f'{result_save_path}/result.pkl', 'rb') as f:
    dataset_matrix_set = pickle.load(f)
val_num = len(dataset_matrix_set[dataset_name]['labels'])
model_ranking_result = {}
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
result_save_path = f'{root_path}/Extracting_information_data/{dataset_name}/split'
print(result_save_path)
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)
with open(f'{result_save_path}/val_split.pkl', 'wb') as f:
    pickle.dump(val_indices_list, f)
with open(f'{result_save_path}/test_split.pkl', 'wb') as f:
    pickle.dump(test_indices_list, f)
