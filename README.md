# MoraBench (**Mo**del **Ra**nking **Bench**mark)

<h1 style="text-align:center">
<img style="vertical-align:middle" width="200" height="200" src="./images/MoraBench_logo.png" />
</h1>

## 🔧 What is it?

**MoraBench** (**Mo**del **Ra**nking **Bench**mark) is a **benchmark platform** comprises a collection of model outputs generated under diverse scenarios. It also provides a **common and easy framework**, for development and evaluation of your own model ranking method within the benchmark.

For more information, checkout our publications:

If you find this repository helpful, feel free to cite our publication:

```
@inproceedings{
    xxx
}
```

## 🔧 What is model ranking?

**Model Ranking** is to rank models from a existing model set according to their performance for target task.

## 🔧 Installation

[1] Install anaconda:
Instructions here: https://www.anaconda.com/download/

[2] Clone the repository:

```
git clone https://github.com/ppsmk388/MoraBench.git
cd MoraBench
```

[3] Create virtual environment:

```
conda env create -f environment.yml
source activate MoraBench
```

<!-- If this not working or you want to use only a subset of modules of Wrench, check out this [wiki page](https://github.com/JieyuZ2/wrench/wiki/Environment-Installation) -->

## 🔧 Available Datasets

The datasets can be downloaded via [this](https://drive.google.com/drive/folders/1_iPhZXG_Vrcgm1Dect3N0iMUZpboYebp?usp=sharing).

A documentation of dataset format and usage can be found in this [wiki-page](https://github.com/ppsmk388/MoraBench/wiki/Detail-of-model-set).

<!-- 

### Weak Supervision:



### Semi-supervised Learning:



### Prompt Selection:

 -->

# 🔧  Quick examples

All example code can be found in [this](https://github.com/ppsmk388/MoraBench/tree/main/examples). For example, for [LEMR framework](https://github.com/ppsmk388/MoraBench/tree/main/examples/LEMR/), we can get its  result by following steps:

### Generate Ranking Correction and Optimal Gap

#### 1. Generate 50 sets of randomized splits for dataset `amazon_review_250_0`:

```sh
python ./morabench/generate_split.py --dataset_name amazon_review_250_0 --split_num 50 
```

#### 2. Calculate the optical gap and ranking correction for different budget ratio for dataset `amazon_review_250_0`:

```sh
python ./examples/LEMR/main.py 
            --Ensemble_method hard              # ensemble method, hard or soft
            --total_split_number 50             # total split number we used
            --dataset_name amazon_review_250_0  # dataset name
            --model_committee_type z_score      # model committee selection type, z_score or all_model
            --seed 0
```

#### 3. Results visualization

```sh
python MoraBench/morabench/plot_result.py --metric rc # rc for ranking correction and og for optimal gap
```

## 🔧  Contact

Contact person: Zhengyu Hu, [huzhengyu477@gmail.com](mailto:huzhengyu477@gmail.com)

Don't hesitate to send us an e-mail if you have any question.

We're also open to any collaboration!

## 🔧  Contributing Dataset and Model

We sincerely welcome any contribution to the methods or model set!

## 🔧  Citattion

```
@inproceedings{
    xxx
}
```
