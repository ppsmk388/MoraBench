import argparse
from morabench.LEMR import lemr_show

parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str,default='rc', help="rc or og")
parser.add_argument("--seed", type=int,default=0, help="seed")
args = parser.parse_args()
metric = args.metric
seed = args.seed
lemr_show(metric_type=metric,seed=seed)
