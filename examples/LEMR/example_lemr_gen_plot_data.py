import argparse
from morabench.utils import setup_seed
from morabench.conifg import root_path
from morabench.LEMR import LEMR
from morabench.dataset import Model_set
ratio_list = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
              0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, ]
Acquisition_list = ['Entropy', 'Uncertainty', 'Margin', 'False', ]
threshold = -2.8
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int,default=0, help="seed")
parser.add_argument("--Ensemble_method", type=str,default='hard', help="Ensemble_method, hard or soft")
parser.add_argument("--total_split_number", type=int,default=50, help="total_split_number")
parser.add_argument("--dataset_name", type=str,default='story', help="dataset_name")
parser.add_argument("--model_committee_type", type=str,default='z_score', help="model_committee_type, z_score or all_model")
args = parser.parse_args()
dataset_name = args.dataset_name
setup_seed(args.seed)
model_hub_path = f'{root_path}/Extracting_information_data/{dataset_name}'
seed = args.seed
Ensemble_method = args.Ensemble_method
total_split_number = args.total_split_number
dataset_name = args.dataset_name
model_committee_type = args.model_committee_type
model_set = Model_set()
model_set.load_model_set(load_path=model_hub_path)
LEMR(Model_set=model_set,
      seed=seed,
      Ensemble_method=Ensemble_method,
      total_split_number=total_split_number,
      dataset_name=dataset_name,
      model_committee_type=model_committee_type)




