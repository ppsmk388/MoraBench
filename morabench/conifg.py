root_path = f'root_path'
unc_method_map = {
    'Entropy': 'Entropy',
    'Uncertainty': "Uncertainty",
    'Margin': "Margin",
    'False': 'Random',
}

ensemble_map={
    'hard':'Hard Ensemble',
    'soft':"Soft Ensemble"
}

usb_dataset_dict = [
                'aclImdb_20_0',
                'aclImdb_100_0',
                'ag_news_40_0',
                'ag_news_200_0',
                'amazon_review_250_0',
                'amazon_review_1000_0',
                'yelp_review_250_0',
                'yelp_review_1000_0',
                'yahoo_answers_500_0',
                'yahoo_answers_2000_0',
                ]


usb_dataset_dict_map = {
                'yelp_review_1000_0': 'Yelp 1000',
                'yelp_review_250_0':"Yelp 250" ,
                'amazon_review_1000_0':"Amazon 1000",
                'amazon_review_250_0':"Amazon 250",
                'aclImdb_100_0':"IMDB 100",
                'aclImdb_20_0': "IMDB 20",
                'ag_news_200_0': "AGNews 200",
                'ag_news_40_0': "AGNews 40",
                'yahoo_answers_500_0': "Yahoo 500",
                'yahoo_answers_2000_0': "Yahoo 2000",
}




prompt_dataset_dict_map = {
                'story': " Story",
                'wsc': "WSC",
                'cb': "CB",
                'rte': 'RTE',
                'wic': "WiC",
                'anli1': "ANLI1",
                'anli2': "ANLI2",
                'anli3': "ANLI3",
}

prompt_dataset_dict = [
                'story',
                'wsc',
                'cb',
                'rte',
                'wic',
                'anli1',
                'anli2',
                'anli3',
                ]


wrench_dataset_dict_map = {
                'yelp_results': "Yelp",
                'sms_results': "SMS",
                'imdb_results':"IMDB" ,
                'ag_results': 'AGNews',
                'trec_results':"Trec",
}


wrench_dataset_dict = [
                'yelp_results',
                'sms_results',
                'imdb_results',
                'ag_results',
                'trec_results',
                ]