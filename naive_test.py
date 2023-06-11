from dataset.molt5_dataset import Mol2CaptionDataset
from transformers import AutoTokenizer
from evaluations.text_translation_metrics import text_evaluate
from evaluations.mol_translation_metrics import mol_evaluate
from evaluations.fingerprint_metrics import molfinger_evaluate
from evaluations.fcd_metric import fcd_evaluate
import argparse


tokenizer = AutoTokenizer.from_pretrained('laituan245/molt5-base-smiles2caption')

parser = argparse.ArgumentParser()
# add raw_folder, pro_folder, dataset_type
parser.add_argument('--raw_folder', type=str, default='./dataset/cap_mol_trans/raw/')
parser.add_argument('--pro_folder', type=str, default='./dataset/cap_mol_trans/llama_1_shot_bm25_morgan/')
parser.add_argument('--dataset_type', type=str, default='test')


args = parser.parse_args()

raw_folder = args.raw_folder
pro_folder = args.pro_folder

test_set = Mol2CaptionDataset(raw_folder, pro_folder, args.dataset_type)


targets = []
preds = []
molecules = []
for i in range(len(test_set)):
    molecules.append(test_set[i][0])
    targets.append(test_set[i][1])
    preds.append(test_set[i][2])

metrics = text_evaluate(tokenizer, targets, preds, molecules, 256)

print('Metrics: bleu-2:{}, bleu-4:{}, rouge-1:{}, rouge-2:{}, rouge-l:{}, meteor-score:{}'.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]))


targets = []
preds = []
descriptions = []

for i in range(len(test_set)):
    descriptions.append(test_set[i][1])
    targets.append(test_set[i][0])
    preds.append(test_set[i][3])

metrics = mol_evaluate(targets, preds, descriptions)
finger_metrics = molfinger_evaluate(targets, preds)
# print(targets[0], preds[0])
fcd_metric= fcd_evaluate(targets, preds)
print("Metrics: bleu_score:{}, em-score:{}, levenshtein:{}, maccs fts:{}, rdk fts:{}, morgan fts:{}, fcd_metric:{}, validity_score:{}".format(metrics[0], metrics[1], metrics[2], finger_metrics[1], finger_metrics[2], finger_metrics[3], fcd_metric, metrics[3]))
