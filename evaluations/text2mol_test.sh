
% Molecule
python write_sdf.py --input_file ../dataset/cap_mol_trans/ten_shot_bm25/test.txt
mol2vec featurize -i tmp.sdf -o tmp.csv -m m2v_model.pkl -r 1 --uncommon UNK
python mol_text2mol_metric.py --input_file ../dataset/cap_mol_trans/ten_shot_bm25/test.txt

% Caption
python text_text2mol_metric.py --input_file ../dataset/cap_mol_trans/ten_shot_morgan/test.txt