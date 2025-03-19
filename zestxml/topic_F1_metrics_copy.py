import os
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pandas as pd
from tabulate import tabulate
from io import StringIO
from tqdm import tqdm
from utils import *
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score
import json


def _filter(score_mat, filter_mat, copy=True):
    if filter_mat is None:
        return score_mat
    if copy:
        score_mat = score_mat.copy()

    temp = filter_mat.tocoo()
    score_mat[temp.row, temp.col] = 0
    del temp
    score_mat = score_mat.tocsr()
    score_mat.eliminate_zeros()
    return score_mat

dataset = sys.argv[1]
dataset_path = sys.argv[2]
RES_DIR = f'{dataset_path}/Results/{dataset}'
# DATA_DIR = f'GZXML-Datasets/{dataset}'
DATA_DIR = f'{dataset_path}'

# print(_c("Loading files", attr="yellow"))
print("Loading files")
trn_X_Y = read_sparse_mat('%s/trn_X_Y.txt'%DATA_DIR, use_xclib=False)
tst_X_Y = read_sparse_mat('%s/tst_X_Y.txt'%DATA_DIR, use_xclib=False)

score_mat = _filter(read_bin_spmat(f'{RES_DIR}/score_mat.bin').copy(), None)
# Shape should be:
# nrows = number of test data
# ncols = scores for possible labels
x = score_mat.toarray()

# getting the set of seen labels in training dataset
seen_labels = set()
train_label = []
with open(f'{DATA_DIR}/trn_X_Y.txt', "r", encoding="utf-8") as re:
    train_label = re.readlines()[1:]
for text in train_label:
    list_labels = []
    split = text.split(" ")
    for label in split:
        label_num = label.split(":")[0]
        seen_labels.add(int(label_num))

#get head, mid, tail topic:
train_text_labels = []
with open(f'{DATA_DIR}/trn_X_Y.txt', "r", encoding="utf-8") as re:
    train_text_labels = re.readlines()[1:]
all_labels = []
for text in train_text_labels:
    split = text.split(" ")
    for label in split:
        label_num = label.split(":")[0]
        all_labels.append(int(label_num))
count_topic = Counter(all_labels)
group_head = [i for i in count_topic if count_topic[i]>=30]
group_mid = [i for i in count_topic if count_topic[i]<30 and count_topic[i]>4]
group_tail = [i for i in count_topic if count_topic[i]<=4]
print("group_head",len(group_head))
print("group_mid", len(group_mid))
print("group_tail", len(group_tail))

# loop through the score matrix
text_labels = []
with open(f'{DATA_DIR}/tst_X_Y.txt', "r", encoding="utf-8") as re:
    text_labels = re.readlines()[1:]
true_labels_index = []
for text in text_labels:
    list_labels = []
    split = text.split(" ")
    for label in split:
        label_num = label.split(":")[0]
        list_labels.append(int(label_num))
    true_labels_index.append(list_labels)

print(len(true_labels_index))
res = []

labels_topk = {"text": [],
               "ground-truth": [],
               "labels_top1": [],
               "labels_top3": [],
               "labels_top5": [],
               "labels_top10":[]}


labels = {}
with open(f'{DATA_DIR}/Yf.txt', "r", encoding="utf-8") as re:
    Yf = re.readlines()
for line in Yf:
    if "__label__" in line:
        id_label = int(line.split("__")[2])
        label = line.split("__")[3][:-2]
        labels[id_label] = label
print(labels[1], labels[664])


for top_k in [1,3,5,10]:
    res_k = []
    print("done1")
    pred_labels = []
    true_labels = []

    for i, rows in enumerate(x):
        predictions = np.argpartition(rows, -top_k)[-top_k:]
        predictions = predictions[::-1]
        labels_topk[f"labels_top{top_k}"].append([labels[int(i)] for i in predictions])
        label = [1 if i in predictions else 0 for i in range(len(rows))]
        pred_labels.append(label)

        label = [1 if j in true_labels_index[i] else 0 for j in range(len(rows))]
        true_labels.append(label)


    # for i, batch in enumerate(test_dataloader):
    #     batch = tuple(t.to(device) for t in batch)
    #     b_input_ids, b_input_mask, b_labels = batch
    #     with torch.no_grad():
    #     # Forward pass
    #         outs = model(b_input_ids, attention_mask=b_input_mask)
    #         b_logit_pred = outs[0]
    #         pred_label = torch.sigmoid(b_logit_pred)
    #         pred_label = pred_label.to('cpu').numpy()
    #         pred_label_all = np.copy(pred_label)
    #         b_labels = b_labels.to('cpu').numpy()

    #         predictions = [np.argsort(row)[-top_k:] for row in pred_label]
    #         predictions = [row[::-1] for row in predictions]

    #         for n in range(len(pred_label)):
    #             for t in range(len(pred_label[n])):
    #                 if t not in predictions[n]:
    #                     pred_label[n][t] = 0

    #     true_labels.append(b_labels)
    #     pred_labels.append(pred_label)
    #     pred_labels_all.append(pred_label_all)
    print("done2")
    # true_labels_test = [item for sublist in true_labels for item in sublist]
    # pred_labels_test = [item for sublist in pred_labels for item in sublist]

    true_labels_test = true_labels 
    pred_labels_test = pred_labels


    best_micro_f1_th = 0




    true_bools = true_labels_test
    print(best_micro_f1_th)
    # print(pred_labels_test_all)
   
    print("done3")
    print(np.array(true_bools).shape)
    print(np.array(pred_labels_test).shape)
    p_test = precision_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_test)],average='micro', zero_division=0)
    re_test = recall_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_test)],average='micro', zero_division=0)
    miF_test = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_test)],average='micro', zero_division=0)
    # mrr_test = mrr_score(true_bools, predictions_all)
    res_k.extend([top_k, best_micro_f1_th, p_test, re_test, miF_test])#, mrr_test])

    # evaluation with same cutoff
    for group_name, group in [('Head',group_head), ('Medium', group_mid), ('Tail',group_tail)]:

        true_bools = [[tl[i]==1 for i in group] for tl in true_labels_test]
        pred_labels_sub = [[pl[i] for i in group] for pl in pred_labels_test]
        print(np.array(true_bools).shape)
        print(np.array(pred_labels_sub).shape)

        p_test = precision_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_sub)],average='micro', zero_division=0)
        re_test = recall_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_sub)],average='micro', zero_division=0)
        miF_test = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_sub)],average='micro', zero_division=0)

        # true_bools = [[1 if i in group and j == 1 else 0 for i,j in enumerate(tl)] for tl in true_labels_test]
        # predictions_subs = [[i for i in tl if i in group] for tl in predictions_all]
        # mrr_test = mrr_score(true_bools, predictions_subs)
        res_k.extend([p_test, re_test, miF_test])#, mrr_test])

    res.append(res_k)
# print(res_model)
# res += res_model.copy()
    
import pandas as pd
print(res)
df_res = pd.DataFrame(res)
df_res.columns = ["Top k", "Threshold", 
                  "Precision-test-All", "Recall-test-All", "F1-test-All",#, "MRR-test-All",
                  "Precision-test-Head", "Recall-test-Head", "F1-test-Head",#, "MRR-test-Head",
                  "Precision-test-Medium", "Recall-test-Medium", "F1-test-Medium",#, "MRR-test-Medium",
                  "Precision-test-Tail", "Recall-test-Tail",  "F1-test-Tail"] #, "MRR-test-Tail"]
df_res.to_excel(f'eval_ZestXML_all_labels.xlsx',index=False)




#+__________read label predict__________+#
# with open(f'{DATA_DIR}/tst_svmlight.txt', "r", encoding="utf-8") as re:
#     test_data = re.readlines()
# print(test_data[0])



test_path = "/home/mybaby/Mydata/NLP4SE-LAB ISE/LEGION/data/ver1/test_data.json"
with open(test_path, 'r') as test_file:
    test_data = json.load(test_file)
for sample in test_data:
    if "askmeegs/learn-istio" in sample["text"] or "niuhuan/jasmine" in sample["text"]:
        continue
    labels_topk["text"].append(sample["text"])
    labels_topk["ground-truth"].append(sample["labels"])

for i in labels_topk:
    print(i, len(labels_topk[i]))
data_predict_zestxml = pd.DataFrame(labels_topk)
# Nếu không muốn index, bạn có thể bỏ index khi lưu vào file CSV như sau:
data_predict_zestxml.to_csv("data_predict_zestxml.csv", index=False)