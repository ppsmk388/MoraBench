

<h1 style="text-align:center">
<img style="vertical-align:middle" width="500" height="180" src="./images/wrench_logo.png" />
</h1>
[![made-with-python](https://img.shields.io/badge/Made%20with-Python3-1f425f.svg?color=purple)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/JieyuZ2/wrench/commits/main)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![repo size](https://img.shields.io/github/repo-size/JieyuZ2/wrench.svg)](https://github.com/JieyuZ2/wrench/archive/master.zip)
[![Total lines](https://img.shields.io/tokei/lines/github/JieyuZ2/wrench?color=red)](https://github.com/JieyuZ2/wrench)
![visitors](https://visitor-badge.glitch.me/badge?page_id=JieyuZ2/wrench)
![GitHub stars](https://img.shields.io/github/stars/JieyuZ2/wrench.svg?color=green)
![GitHub forks](https://img.shields.io/github/forks/JieyuZ2/wrench?color=9cf)
[![Arxiv](https://img.shields.io/badge/ArXiv-2109.11377-orange.svg)](https://arxiv.org/abs/2109.11377) 













# MoraBench (**Mo**del **Ra**nking **Bench**mark)


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


A documentation of dataset format and usage can be found in this [wiki-page](https://github.com/ppsmk388/MoraBench/wiki/Dataset:-Format-and-Usage)

<!-- 
### Weak Supervision:



### Semi-supervised Learning:


### Prompt Selection:


 -->



# 🔧  Quick examples





### Generate Ranking Correction and Optimal Gap
```sh
python main.py --dataset_name story
```
###   Results visualization

```sh
python plot_result.py --metric rc
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



