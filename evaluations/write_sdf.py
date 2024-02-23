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


import pickle
import argparse
import csv

import os.path as osp


from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()
parser.add_argument('--raw_file', type=str, default='../dataset/cap_mol_trans/raw/', help='path where data is located')
parser.add_argument('--input_file', type=str, default='../dataset/cap_mol_trans/ten_shot_bm25/test.txt', help='path where test generations are saved')


args = parser.parse_args()

with open('cid_to_smiles.pkl', 'rb') as f:
    cids_to_smiles = pickle.load(f)

#remove '*' smiles from dataset
cids_to_remove = []
for cid in cids_to_smiles:
    if cids_to_smiles[cid] == '*':
        cids_to_remove.append(cid)

for cid in cids_to_remove:
    cids_to_smiles.pop(cid, None)


for cid in cids_to_smiles:
    if cids_to_smiles[cid] == '*': continue
    m = Chem.MolFromSmiles(cids_to_smiles[cid])
    smi = Chem.MolToSmiles(m)
    cids_to_smiles[cid] = smi


smiles_to_cids = {}
smiles_to_cids['*'] = []

for cid in cids_to_smiles:
    if cids_to_smiles[cid] in smiles_to_cids: print(cid)
    
    if cids_to_smiles[cid] == '*':
        smiles_to_cids[cids_to_smiles[cid]].append(cid)
        continue

    smiles_to_cids[cids_to_smiles[cid]] = cid

outputs = []

#load data
output_descs = []
output_gts = []
with open(osp.join(args.raw_file, 'test'+'.txt')) as f:
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for n, line in enumerate(reader):
        output_descs.append(line['description'])
        gt_smi = line['SMILES']
        m = Chem.MolFromSmiles(gt_smi)
        gt_smi = Chem.MolToSmiles(m)
        output_gts.append(gt_smi)

bad_mols = 0
output_ots = []
with open(osp.join(args.input_file)) as f:
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for n, line in enumerate(reader):
        try:
            ot_smi = line['SMILES']
            if ot_smi == '': #fixes a downstream error in mol2vec
                raise ValueError('Empty molecule.')
            m = Chem.MolFromSmiles(ot_smi)
            ot_smi = Chem.MolToSmiles(m)
            output_ots.append(m)
        except:
            bad_mols += 1

outputs = list(zip(output_descs, output_gts, output_ots))
print('validity:', len(outputs)/(len(outputs)+bad_mols))

with Chem.SDWriter('tmp.sdf') as w:
    for o in outputs:
        m = o[2]
        m.SetProp("CID", smiles_to_cids[o[1]])
        w.write(m)