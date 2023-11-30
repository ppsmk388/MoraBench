# MoraBench (**Mo**del **Ra**nking **Bench**mark)

## Code

This is the official implementation of [*How Many Validation Labels Do You Need? Exploring the Design Space of Label-Efficient Model Ranking*]() 

## Environment

* Python=3.8.18
* PyTorch=1.10.1
* numpy=1.20.3

## Usage

### Datasets

Our data has been preprocessed and is available at 





### Generate Ranking Correction and Optimal Gap
```sh
python main.py --dataset_name story
```
###   Results visualization

```sh
python plot_result.py --metric rc
```
