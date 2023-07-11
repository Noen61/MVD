# MVD: Towards better Entity Linking with Multi-View enhanced Distillation

This repo provides the code for our ACL 2023 paper: **[Towards better Entity Linking with Multi-View Enhanced Distillation](https://aclanthology.org/2023.acl-long.542.pdf)**

## Setup

Run the following command to install the required packages:

```
pip install -r requirements.txt
```

The code for our paper was run using 4 NVIDIA RTX A6000

## Data Download & Preprocess

For **ZESEHL** dataset:

```
sh scripts/get_zeshel_data.sh
```

```
python utils/preprocess_data.py --dataset zeshel --mention_path data/zeshel/mentions --document_path data/zeshel/documents --output_path data/zeshel --max_seq_length 128 --max_ent_length 512 --max_view_length 40
```

For **Wikipedia-based** datasets:

```
wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
```

```
sh scripts/get_wikipedia_data.sh
```

```
python utils/preprocess_data.py --dataset wikipedia \
--mention_path data/wikipedia/basic_data/test_datasets \
--document_path data/wikipedia/kilt_knowledgesource.json \
--output_path data/wikipedia/entity.json \
--max_seq_length 32 --max_ent_length 512 --max_view_length 40
```

## Warmup Training: Retriever

For **ZESEHL** dataset:

Run the follow script to train the retriver or you can download the warmup-trained retriever [here](https://drive.google.com/file/d/1oYyfzq5kDNWZF502X2vt9fYKEfSjUjJ_/view?usp=sharing)

```
sh scripts/run_retriever.sh
```

Computational cost:

    Time:  around 0.6h,  1min per epoch

    Memory: 20G per GPU

For **Wikipedia-based** Datasets: change several **hyperparameters**

```
--kb wikipedia 
--data_dir data/wikipedia
--entity_data_dir data/wikipedia/entity.json
--output_dir output/wikipedia
--bert_model bert-large-uncased
--max_seq_length 32
--max_view_num 5 
```

## Generate Hard Negatives

Run the follow script to generate **static hard negatives** for the teacher model:

```
sh scripts/run_predict.sh
```

## Warmup Training: Teacher

Run the follow script to train the teacher or you can download the warmup-trained teacher [here](https://drive.google.com/file/d/1MPWiCnTjE_wTGYrGe7DAPTemjonKs83Z/view?usp=sharing)

```
sh scripts/run_teacher.sh
```

Computational cost:

    Time:  around 2h,  40min per epoch

    Memory: 25G per GPU

## MVD Training

Run the follow script to MVD or you can download the retriever after MVD training [here](https://drive.google.com/file/d/17DOtfKwSCjS9kZsDFG0lQQ0kghXXtE1u/view?usp=sharing)

```
sh scripts/run_mvd.sh
```

Computational cost:

    Time:   around 13h,  2.5h per epoch

    Memory: 35G per GPU

## Evaluation

Run the follow script to evaluate the retrieval performance:

```
sh scripts/run_evaluate.sh
```

### Retrieval results on ZESHEL dataset

| Model                 |    Recall@1    | Recall@2        | Recall@4        | Recall@8        | Recall@16       | Recall@32       | Recall@50       | Recall@64       |
| :-------------------- | :-------------: | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| Warmup                |      41.99      | 57.54           | 68.06           | 75.28           | 81.04           | 85.74           | 88.20           | 89.23           |
| MVD (w/o global-view) |      51.27      | 63.30           | 71.81           | 78.04           | 82.99           | 86.81           | 89.05           | 90.25           |
| MVD                   | **52.51** | **64.77** | **73.43** | **79.74** | **84.35** | **88.17** | **90.43** | **91.55** |

### Retrieval results on Wikipedia-based datasets

Recall@1 of MVD is even higher than overall accuracy (retrieval-then-rank) of [BLINK](https://github.com/facebookresearch/BLINK)

#### AIDA-testb

| Model                                           |    Recall@1    |    Recall@10    |    Recall@30    |   Recall@100   | Overall Acc |
| :---------------------------------------------- | :-------------: | :-------------: | :-------------: | :-------------: | :---------: |
| [BLINK](https://github.com/facebookresearch/BLINK) |      79.51      |      92.38      |      94.87      |      96.63      |    80.27    |
| [MuVER](https://github.com/Alibaba-NLP/MuVER)      |        -        |      94.53      |      95.25      |      98.11      |      -      |
| MVD (w/o global-view)                           | **84.32** | **97.05** | **98.15** | **98.80** |      -      |

#### MSNBC

| Model                                           |    Recall@1    |    Recall@10    |    Recall@30    |   Recall@100   | Overall Acc |
| :---------------------------------------------- | :-------------: | :-------------: | :-------------: | :-------------: | :---------: |
| [BLINK](https://github.com/facebookresearch/BLINK) |      84.28      |      93.03      |      95.46      |      96.76      |    85.09    |
| [MuVER](https://github.com/Alibaba-NLP/MuVER)      |        -        |      95.02      |      96.62      |      97.75      |      -      |
| MVD (w/o global-view)                           | **85.66** | **96.74** | **97.71** | **98.04** |      -      |

#### WNED-CWEB

| Model                                           |    Recall@1    |    Recall@10    |    Recall@30    |   Recall@100   | Overall Acc |
| :---------------------------------------------- | :-------------: | :-------------: | :-------------: | :-------------: | :---------: |
| [BLINK](https://github.com/facebookresearch/BLINK) |      67.07      |      82.23      |      86.09      |      88.68      |    68.28    |
| [MuVER](https://github.com/Alibaba-NLP/MuVER)      |        -        |      79.31      |      83.94      |      88.15      |      -      |
| MVD (w/o global-view)                           | **68.87** | **85.01** | **88.18** | **91.11** |      -      |

## Acknowledgements

We thank [BLINK](https://github.com/facebookresearch/BLINK) and [MuVER](https://github.com/Alibaba-NLP/MuVER) for the base infrastructure of this project.
