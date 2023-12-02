import copy
import pickle
from morabench.utils import  gen_ensemble_matrix,gen_ensemble_result_from_matrix


class Model_set():
    def __init__(self):
        self.dataset_matrix_set = None
        self.Model_result_dict = {}
        self.result_distribution = {}



    def load_model_set(self, load_path):
        with open(load_path, 'rb') as file:
            self.dataset_matrix_set = pickle.load(file)

    def gen_distribute(self,dataset_name, Ensemble_method):
        Soft_ensemble_dict = copy.deepcopy(self.dataset_matrix_set[dataset_name]['Soft_ensemble_dict'])
        Hard_ensemble_dict = copy.deepcopy(self.dataset_matrix_set[dataset_name]['Hard_ensemble_dict'])
        Soft_ensemble_matrix = gen_ensemble_matrix(Soft_ensemble_dict)
        Hard_ensemble_matrix = gen_ensemble_matrix(Hard_ensemble_dict)
        _, Soft_ensemble_result_distribution = gen_ensemble_result_from_matrix(Soft_ensemble_matrix)
        _, Hard_ensemble_result_distribution = gen_ensemble_result_from_matrix(Hard_ensemble_matrix)

        if Ensemble_method == 'hard':
            self.Model_result_dict[dataset_name] = copy.deepcopy(Hard_ensemble_dict)
            self.result_distribution[dataset_name] = copy.deepcopy(Hard_ensemble_result_distribution)
        else:
            self.Model_result_dict[dataset_name] = copy.deepcopy(Soft_ensemble_dict)
            self.result_distribution[dataset_name] = copy.deepcopy(Soft_ensemble_result_distribution)



