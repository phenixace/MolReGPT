# MolReGPT: Empowering Molecule Discovery for Molecule-Caption Translation with Large Language Models: A ChatGPT Perspective

The official repo of `MolReGPT` - **"Empowering Molecule Discovery for Molecule-Caption Translation with Large Language Models: A ChatGPT Perspective"**. (To appear in Arxiv) 

#### Author List
| Name | Affiliation |
| :---: | :---: |
| Jiatong Li | The Hong Kong Polytechnic University |
| Yunqing Liu | The Hong Kong Polytechnic University |
| Wenqi Fan | The Hong Kong Polytechnic University |
| Xiao-Yong Wei | The Hong Kong Polytechnic University & Sichuan University |
| Hui Liu | Michigan State University |
| Jiliang Tang | Michigan State University |
| Qing Li | The Hong Kong Polytechnic University |

#### Model Strcuture
[![model](./figs/model_structure.png)](./figs/model_structure.png)

#### Contact
If you encounter any problems, please feel free to contact us via [email](jiatong.li@connect.polyu.hk)


## News
Codes, data, and demo will be available soon~ We will follow our roadmap to release the results. Please stay tuned!

#### In the first phase:
We will first release the results of [zero_shot](./dataset/cap_mol_trans/zero_shot/). 

The remaining results will be gradually released. Thanks for your patience!

#### In the second phase and later:

Secret Now! 🤫 We will release our plan~ 😄

## Introduction
MolReGPT aims to create a foundation method for molecule discovery by leveraging large language models (LLMs). 
Thus, we focus on two crucial aspects: 
1. molecule understanding
2. text-conditioned molecule generation 

To this end, we focus on a specific task, `molecule-caption translation`, the two sub-tasks of which exactly corresponds to the two aspects. 
1. `molecule2caption (i.e., Mol2Cap)` aims to generate a caption for a given molecule to describe its structure, properties, and functions.
2. `caption2molecule (i.e., Cap2Mol)` aims to generate a molecule for a given caption, which could help researchers customize their molecules for specific purposes.


### Dataset
We apply the same dataset used in MolT5, which is the dataset of [ChEBI-20](./dataset/cap_mol_trans/raw/)

### Results

Results will be released once the paper is public to the community (scheduled 2023/06/13:00:00:00 GMT). Thanks for you attention!

#### Mol2Cap
| Method | BLEU-2 $\uparrow$ | BLEU-4 $\uparrow$| ROUGEL-1 $\uparrow$| ROUGEL-2 $\uparrow$ | ROUGEL-L $\uparrow$ | METEOR $\uparrow$ | Text2Mol $\uparrow$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Transformer | 0.061 | 0.027 | 0.204 | 0.087 | 0.186 | 0.114 | 0.057 |
| T5-base | 0.511 | 0.423 | 0.607 | 0.451 | 0.550 | 0.539 | 0.523 |
| MolT5-base | 0.540 | 0.457 | 0.634 | 0.485 | 0.578 | 0.569 | 0.547 |
| GPT-3.5-turbo (zero_shot) | 0.103 | 0.050 | 0.261 | 0.088 | 0.204 | 0.161 | 0.352 |
| MolReGPT |


#### Cap2Mol
| Method | BLEU $\uparrow$ | EM $\uparrow$ | Levenshtein $\downarrow$ | MACCS FTS $\uparrow$ | RDK FTS $\uparrow$ | Morgan FTS $\uparrow$  | FCD $\downarrow$ | Text2Mol $\uparrow$ | VAlidity $\uparrow$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Transformer | 0.499 | 0.000 | 57.66 | 0.480 | 0.320 | 0.217 | 11.32 | 0.277 | 0.906 |
| T5-base | 0.762 | 0.069 | 24.950 | 0.731 | 0.605 | 0.545 | 2.48 | 0.499 | 0.660 |
| MolT5-base | 0.769 | 0.081 | 24.458 | 0.721 | 0.588 | 0.529 | 2.18 | 0.496 | 0.772|
| GPT-3.5-turbo (zero_shot) | 0.489 | 0.019 | 52.13 | 0.705 | 0.462 | 0.367 | 2.05 | 0.479 | 0.802 |
| MolReGPT |


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
