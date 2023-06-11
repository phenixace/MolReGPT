# MolReGPT

The official repo of MolReGPT - Empowering Molecule Discovery for Molecule-Caption Translation with Large Language Models: A ChatGPT Perspective. (To appear in Arxiv) 

Codes, data, and demo will be available soon~

We will first release the [zero_shot](./dataset/cap_mol_trans/zero_shot/). 
The remaining results will be gradually released.

If you encounter any problems, please feel free to contact us via [email](jiatong.li@connect.polyu.hk)

## Introduction
MolReGPT aims to create a foundation method for molecule discovery by leveraging large language models (LLMs).
### Model Strcuture

TBA

### Dataset
We apply the same dataset used in MolT5, which is the dataset of [ChEBI-20](./dataset/cap_mol_trans/raw/)

### Results

Results will be released once the paper is public to the community (scheduled 2023/06/13:00:00:00 GMT). Thanks for you attention!

#### Mol2Cap

#### Cap2Mol

## Requirements

```
transformers == 4.30.0
torch == 1.13.1+cu117
rdkit == 2022.09.5
fcd == 1.1
rank_bm25 == 0.2.2
sentence_transformers == 2.2.2
openai == 0.27.2
```

## Usage

TBA

## Demo

Currently, you can customize your own prompt via our released [jupyter-notebook demo](./inference.ipynb)

We will also provide web-page for a more user-friendly demo.

## Citation
```
@inproceedings{molregpt,
  title={Empowering Molecule Discovery for Molecule-Caption Translation with Large Language Models: A ChatGPT Perspective},
  author={Jiatong Li and Yunqing Liu and Wenqi Fan and Xiao-Yong Wei and Hui Liu and Jiliang Tang and Qing Li},
  booktitle={Arxiv},
  year={2023 (to appear)}
}
```
