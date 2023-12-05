#!/bin/bash


# Set the default value to 50
total_split_number=50

# If the command line parameter is given, use that
if [ ! -z "$1" ]; then
    total_split_number=$1
fi

for Ensemble_method in hard soft
do
    for dataset_name in  story  wsc cb rte wic anli1 anli2 anli3
    do
        for model_committee_type in z_score all_model
        do
            echo run_lemr.py --Ensemble_method $Ensemble_method --dataset_name $dataset_name --total_split_number $total_split_number   --total_split_number $model_committee_type
            python run_lemr.py --Ensemble_method $Ensemble_method --dataset_name $dataset_name --total_split_number $total_split_number   --total_split_number $model_committee_type
        done
    done
done





