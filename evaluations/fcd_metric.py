'''
Code from https://github.com/blender-nlp/MolT5
```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''

import argparse
import csv

import os.path as osp

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from fcd import get_fcd, load_ref_model, canonical_smiles

def fcd_evaluate(targets, preds, verbose=False):


    model = load_ref_model()

    canon_gt_smis = [w for w in canonical_smiles(targets)]
    canon_ot_smis = [w for w in canonical_smiles(preds)]
    # print(canon_gt_smis[0], canon_ot_smis[0])
    fcd_sim_score = get_fcd(canon_gt_smis, canon_ot_smis, model)
    if verbose:
        print('FCD Similarity:', fcd_sim_score)

    return fcd_sim_score
