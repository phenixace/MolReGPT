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

import numpy as np

import torch

from text2mol.code.models import MLPModel

from transformers import BertTokenizerFast

from sklearn.metrics.pairwise import cosine_similarity

from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()

parser.add_argument('--use_gt', action=argparse.BooleanOptionalAction)

parser.add_argument('--input_file', type=str, default='../dataset/cap_mol_trans/ten_shot_morgan/test.txt', help='path where test generations are saved')

parser.add_argument('--raw_file', type=str, default='../dataset/cap_mol_trans/raw/', help='raw_file')

parser.add_argument('--data_path', type=str, default='text2mol_data/', help='path where data is located.')

parser.add_argument('--split', type=str, default='test', help='data split to evaluate text2mol with.')

parser.add_argument('--text_model', type=str, default='allenai/scibert_scivocab_uncased', help='Desired language model.')

parser.add_argument('--checkpoint', type=str, default='t2m_output/test_outputfinal_weights.320.pt', help='Text2Mol checkpoint to use.')

parser.add_argument('--text_trunc_length', type=str, default=256, help='tokenizer maximum length')


args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

mol2vec = {}
with open(osp.join(args.data_path, args.split+'.txt')) as f:
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'mol2vec', 'desc'])
    for n, line in enumerate(reader):
        mol2vec[line['cid']] = np.fromstring(line['mol2vec'], sep=" ")

#load data
output_smi = []
output_gts = []
with open(osp.join(args.raw_file, args.split+'.txt')) as f:
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for n, line in enumerate(reader):
        # print(line['SMILES'])
        
        output_gts.append(line['description'])
        m = Chem.MolFromSmiles(line['SMILES'])
        smi = Chem.MolToSmiles(m)
        if smi == "[C]":
            smi = "C"
        output_smi.append(smi)
# print(mol2vec)
output_ots = []
with open(osp.join(args.input_file)) as f:
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for n, line in enumerate(reader):
        out_tmp = line['molecule2caption']
        output_ots.append(out_tmp)

outputs =list(zip(output_smi, output_gts, output_ots))



text_tokenizer = BertTokenizerFast.from_pretrained(args.text_model)

model = MLPModel(ninp = 768, nhid = 600, nout = 300)

tmp = model.to(device)

model.load_state_dict(torch.load(args.checkpoint))

model.eval()

sims = []

mol_embs = []
text_embs = []

with torch.no_grad():
    for i, (smi, gt, out) in enumerate(outputs):
        
        if i % 100 == 0: print(i, 'processed.')

        cid = smiles_to_cids[smi]
        
        if args.use_gt: text = gt
        else: text = out

        m2v = mol2vec[cid]

        #print(text)
        text_input = text_tokenizer(text, truncation=True, max_length=args.text_trunc_length,
                                            padding='max_length', return_tensors = 'pt')

        input_ids = text_input['input_ids'].to(device)
        attention_mask = text_input['attention_mask'].to(device)
        molecule = torch.from_numpy(m2v).reshape((1,300)).to(device).float()

        text_emb, mol_emb = model(input_ids, molecule, attention_mask)

        text_emb = text_emb.cpu().numpy()
        mol_emb = mol_emb.cpu().numpy()

        text_embs.append(text_emb)
        mol_embs.append(mol_emb)

        sims.append(abs(cosine_similarity(text_emb, mol_emb)[0][0]))
        

print('Average Similarity:', np.mean(sims))

text_embs = np.array(text_embs).squeeze()
mol_embs = np.array(mol_embs).squeeze()

mat = cosine_similarity(text_embs, mol_embs)
print('Negative Similarity:', np.mean(mat[np.eye(mat.shape[0]) == 0]))