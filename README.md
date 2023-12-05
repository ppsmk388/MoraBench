# MoraBench (**Mo**del **Ra**nking **Bench**mark)

<h1 style="text-align:center">
<img style="vertical-align:middle" width="200" height="200" src="./images/MoraBench_logo.png" />
</h1>

## 🤔 What is it?

**MoraBench** (**Mo**del **Ra**nking **Bench**mark) is a **benchmark platform** comprises a collection of model outputs generated under diverse scenarios. It also provides a **common and easy framework**, for development and evaluation of your own model ranking method within the benchmark.

## 🏁 What is model ranking?

**Model Ranking** is to rank models from a set of **trained models** according to their performance for the target task. Traditionally, people use a **fully-labeled validation set** to rank the models, here we explore how to do model ranking with limited annotation budget.

## 🔧 Installation

### Using conda

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

### Using pip

```
pip install -r requirements.txt
```

<!-- If this not working or you want to use only a subset of modules of Wrench, check out this [wiki page](https://github.com/JieyuZ2/wrench/wiki/Environment-Installation) -->


## 📊 Available Model-sets

MoraBench assembles outputs from models operating under different learning paradigms:


### Weak Supervision:

We generated model outputs within a weak supervision setting using the [WRENCH](https://github.com/JieyuZ2/wrench) framework. We generate model outputs across 48 distinct weak supervision configurations on five datasets: SMS, AGNews, Yelp, IMDB, Trec.



### Semi-supervised Learning:

Leveraging the [USB](http://github.com/microsoft/Semi-supervised-learning) benchmark, model outputs were obtained from 12 semi-supervised methods across five datasets: IMDB, Amazon Review, Yelp Review, AGNews and Yahoo! Answer. 




### Prompt Selection:

We employed large language models and various prompts to generate diverse outputs, assessed using the [T0](http://github.com/bigscience-workshop/T0) benchmark.






The table below shows the initial model set included in MoraBench and the total size of the validation set plus the test set, i.e., **\# Data**.  The number after the dataset of Semi-supervised Learning indicates the number of labels used in semi-supervised training stage.


|     Training Setting     	|            Task Type        	|        Dataset       	| Model Number 	| # Data 	|
|:------------------------:	|:--------------------------:	|:--------------------:	|:------------:	|:------:	|
|     Weak Supervision     	|  Sentiment Classification  	|         Yelp         	|      480     	|  3800  	|
|                          	|  Sentiment Classification  	|         IMDB         	|      480     	|  2500  	|
|                          	|     Spam Classification    	|          SMS         	|      480     	|   500  	|
|                          	|     Spam Classification    	|         IMDB         	|      480     	|  2500  	|
|                          	|    Topic Classification    	|        AGNews        	|      159     	|  12000 	|
|                          	|   Question Classification  	|         Trec         	|      45      	|   500  	|
|                          	|                            	|                      	|              	|        	|
|                          	|                            	|                      	|              	|        	|
| Semi-supervised Learning 	|  Sentiment Classification  	|       IMDB (20)      	|      400     	|  2000  	|
|                          	|  Sentiment Classification  	|      IMDB (100)      	|      400     	|  2000  	|
|                          	|  sentiment classification  	|   Yelp Review (250)  	|      400     	|  25000 	|
|                          	|  sentiment classification  	|  Yelp Review (1000)  	|      400     	|  25000 	|
|                          	|  sentiment classification  	|  Amazon Review (250) 	|      400     	|  25000 	|
|                          	|  Sentiment Classification  	| Amazon Review (1000) 	|      400     	|  25000 	|
|                          	|    Topic Classification    	|  Yahoo! Answer (500) 	|      400     	|  50000 	|
|                          	|    Topic Classification    	| Yahoo! Answer (2000) 	|      400     	|  50000 	|
|                          	|    Topic Classification    	|      AGNews (40)     	|      400     	|  10000 	|
|                          	|    Topic Classification    	|     AGNews (200)     	|      400     	|  10000 	|
|                          	|                            	|                      	|              	|        	|
|                          	|                            	|                      	|              	|        	|
|     Prompt Selection     	|   Coreference Resolution   	|          WSC         	|      10      	|   104  	|
|                          	|  Word Sense Disambiguation 	|          WiC         	|      10      	|   638  	|
|                          	|     Sentence Completion    	|         Story        	|       6      	|  3742  	|
|                          	| Natural Language Inference 	|          CB          	|      15      	|   56   	|
|                          	| Natural language Inference 	|          RTE         	|      10      	|   277  	|
|                          	| Natural language Inference 	|         ANLI1        	|      15      	|  1000  	|
|                          	| Natural language Inference 	|         ANLI2        	|      15      	|  1000  	|
|                          	| Natural language Inference 	|         ANLI3        	|      15      	|  1200  	|



Details of these datasets can be found in our [paper](https://arxiv.org/abs/2312.01619), and all these model sets can be downloaded via [this](https://drive.google.com/drive/folders/1_iPhZXG_Vrcgm1Dect3N0iMUZpboYebp?usp=sharing). We plan to add more model set soon.









# 📙  Quick examples

All example code can be found in [this](https://github.com/ppsmk388/MoraBench/tree/main/examples). 
For example, for [LEMR framework](https://github.com/ppsmk388/MoraBench/tree/main/examples/LEMR/), we can show its result of prompt selection setting by following steps:





### 1. Generate plot data:

We can directly run . /examples/run.sh

```sh
bash ./examples/LEMR/run.sh num_split
```

where num_split is the number of splits generated and if not entered, the default is 50.

Here are the details of `run.sh`

```sh
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
            python split_data_merge_all_model.py 
            --Ensemble_method $Ensemble_method              # ensemble method, hard or soft
            --dataset_name $dataset_name                    # dataset name
            --total_split_number $total_split_number        # total split number we used
            --total_split_number $model_committee_type      # model committee selection type, z_score or all_model
        done
    done
done

```

<!-- #### 2. Calculate the optical gap and ranking correction for different budget ratio for dataset `amazon_review_250_0`:

```sh
python ./examples/LEMR/main.py 
            --Ensemble_method hard              # ensemble method, hard or soft
            --total_split_number 50             # total split number we used
            --dataset_name amazon_review_250_0  # dataset name
            --model_committee_type z_score      # model committee selection type, z_score or all_model
            --seed 0
``` -->

### 2. Results visualization

```sh
python ./examples/LEMR/lemr_show_result.py --metric rc # rc for ranking correction and og for optimal gap
```



## 📧  Contact

Contact person: Zhengyu Hu, [huzhengyu477@gmail.com](mailto:huzhengyu477@gmail.com)

Don't hesitate to send us an e-mail if you have any question.

We're also open to any collaboration!

## ✨  Contributing Dataset and Model

We sincerely welcome any contribution to the methods or model set!

## 👆  Citattion

```
@inproceedings{Hu2023HowMV,
  title={How Many Validation Labels Do You Need? Exploring the Design Space of Label-Efficient Model Ranking},
  author={Zhengyu Hu and Jieyu Zhang and Yue Yu and Yuchen Zhuang and Hui Xiong},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:265610019}
}
```
