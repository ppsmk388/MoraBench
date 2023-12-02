import argparse
from morabench.LEMR import LEMR_plot


parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str,default='rc', help="rc or og")
parser.add_argument("--seed", type=int,default=0, help="seed")
args = parser.parse_args()
metric = args.metric
seed = args.seed
LEMR_plot(metric_type=metric,seed=seed)
