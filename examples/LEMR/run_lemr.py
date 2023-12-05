import argparse
from morabench.utils import setup_seed
from morabench.conifg import root_path
from morabench.LEMR import LEMR

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int,default=0, help="seed")
parser.add_argument("--Ensemble_method", type=str,default='hard', help="Ensemble_method, hard or soft")
parser.add_argument("--total_split_number", type=int,default=50, help="total_split_number")
parser.add_argument("--dataset_name", type=str,default='story', help="dataset_name")
parser.add_argument("--model_committee_type", type=str,default='z_score', help="model_committee_type, z_score or all_model")
args = parser.parse_args()

dataset_name = args.dataset_name
seed = args.seed
Ensemble_method = args.Ensemble_method
total_split_number = args.total_split_number
model_committee_type = args.model_committee_type
setup_seed(args.seed)
model_hub_path = f'{root_path}/Extracting_information_data/{dataset_name}'

if model_committee_type == 'z_score':
      result_save_path = f'{root_path}/Extracting_information_data/dataset_name_{dataset_name}/' \
                         f'Ensemble_method_{Ensemble_method}/seed_{seed}/'
else:
      result_save_path = f'{root_path}/Extracting_information_data/all_model/dataset_name_{dataset_name}/' \
                         f'Ensemble_method_{Ensemble_method}/seed_{seed}/'
lemr = LEMR(model_set_path=model_hub_path,
            seed=seed,
            Ensemble_method=Ensemble_method,
            total_split_number=total_split_number,
            dataset_name=dataset_name,
            model_committee_type=model_committee_type,
            result_save_path=result_save_path

            )

lemr.rank()


