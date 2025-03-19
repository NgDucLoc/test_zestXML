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
actuals = []
for text in text_labels:
    list_labels = []
    split = text.split(" ")
    for label in split:
        label_num = label.split(":")[0]
        list_labels.append(int(label_num))
    actuals.append(list_labels)
sum_sr_1 = 0
sum_sr_2 = 0
sum_sr_3 = 0
sum_sr_4 = 0
sum_sr_5 = 0
sum_sr_6 = 0
sum_sr_7 = 0
sum_sr_8 = 0
sum_sr_9 = 0
sum_sr_10 = 0
sum_recall_1 = 0
sum_recall_2 = 0
sum_recall_3 = 0
sum_recall_4 = 0
sum_recall_5 = 0
sum_recall_6 = 0
sum_recall_7 = 0
sum_recall_8 = 0
sum_recall_9 = 0
sum_recall_10 = 0
sum_precision_1 = 0
sum_precision_2 = 0
sum_precision_3 = 0
sum_precision_4 = 0
sum_precision_5 = 0
sum_precision_6 = 0
sum_precision_7 = 0
sum_precision_8 = 0
sum_precision_9 = 0
sum_precision_10 = 0
num_test_data = 0
total_labels = 0
total_labels_head = 0
total_labels_mid = 0
total_labels_tail = 0

sum_sr_1_head = 0
sum_sr_2_head = 0
sum_sr_3_head = 0
sum_sr_4_head = 0
sum_sr_5_head = 0
sum_sr_6_head = 0
sum_sr_7_head = 0
sum_sr_8_head = 0
sum_sr_9_head = 0
sum_sr_10_head = 0
sum_recall_1_head = 0
sum_recall_2_head = 0
sum_recall_3_head = 0
sum_recall_4_head = 0
sum_recall_5_head = 0
sum_recall_6_head = 0
sum_recall_7_head = 0
sum_recall_8_head = 0
sum_recall_9_head = 0
sum_recall_10_head = 0
sum_precision_1_head = 0
sum_precision_2_head = 0
sum_precision_3_head = 0
sum_precision_4_head = 0
sum_precision_5_head = 0
sum_precision_6_head = 0
sum_precision_7_head = 0
sum_precision_8_head = 0
sum_precision_9_head = 0
sum_precision_10_head = 0

sum_sr_1_mid = 0
sum_sr_2_mid = 0
sum_sr_3_mid = 0
sum_sr_4_mid = 0
sum_sr_5_mid = 0
sum_sr_6_mid = 0
sum_sr_7_mid = 0
sum_sr_8_mid = 0
sum_sr_9_mid = 0
sum_sr_10_mid = 0
sum_recall_1_mid = 0
sum_recall_2_mid = 0
sum_recall_3_mid = 0
sum_recall_4_mid = 0
sum_recall_5_mid = 0
sum_recall_6_mid = 0
sum_recall_7_mid = 0
sum_recall_8_mid = 0
sum_recall_9_mid = 0
sum_recall_10_mid = 0
sum_precision_1_mid = 0
sum_precision_2_mid = 0
sum_precision_3_mid = 0
sum_precision_4_mid = 0
sum_precision_5_mid = 0
sum_precision_6_mid = 0
sum_precision_7_mid = 0
sum_precision_8_mid = 0
sum_precision_9_mid = 0
sum_precision_10_mid = 0

sum_sr_1_tail = 0
sum_sr_2_tail = 0
sum_sr_3_tail = 0
sum_sr_4_tail = 0
sum_sr_5_tail = 0
sum_sr_6_tail = 0
sum_sr_7_tail = 0
sum_sr_8_tail = 0
sum_sr_9_tail = 0
sum_sr_10_tail = 0
sum_recall_1_tail = 0
sum_recall_2_tail = 0
sum_recall_3_tail = 0
sum_recall_4_tail = 0
sum_recall_5_tail = 0
sum_recall_6_tail = 0
sum_recall_7_tail = 0
sum_recall_8_tail = 0
sum_recall_9_tail = 0
sum_recall_10_tail = 0
sum_precision_1_tail = 0
sum_precision_2_tail = 0
sum_precision_3_tail = 0
sum_precision_4_tail = 0
sum_precision_5_tail = 0
sum_precision_6_tail = 0
sum_precision_7_tail = 0
sum_precision_8_tail = 0
sum_precision_9_tail = 0
sum_precision_10_tail = 0

prediction_not_seen = {}
prediction_not_seen_correct = {}

prediction_all_check = {}

print(len(x[0]))



for i, rows in enumerate(x):
    predictions = np.argpartition(rows, -10)[-10:]
    predictions = predictions[::-1]
    # print("predictions: ",predictions)
    temp_pred = rows[predictions[:10]]
    new_sort = np.argsort(temp_pred)
    ns = list(new_sort)
    # print("ns", ns)
    
    num_test_data += 1
    local_correct_prediction = 0
    labels = actuals[i]
    # get only the top k prediction
    # print("labels: ", labels)

    for label in labels:
        if label in group_head:
            total_labels_head += 1
        elif label in group_mid:
            total_labels_mid += 1
        elif label in group_tail:
            total_labels_tail += 1
    total_labels += len(labels)
    correct_prediction = 0
    correct_prediction_head = 0
    correct_prediction_mid = 0
    correct_prediction_tail = 0
    
    # print("Prediction labels")
    # print("SCORES")
    # print(temp_pred)
    # print("BEFORE")
    # print(predictions)
    # print("AFTER")
    # print(new_sort)

    
    # prediction = predictions[predictions != 0]
    # print(labels)
    # labels.remove(0)


    # print([predictions[ns[len(ns)-1]],predictions[ns[len(ns)-2]],predictions[ns[len(ns)-3]]])
    # # print(type(predictions))
    # print(labels)
    # print()
    if predictions[ns[len(ns)-1]] in labels:
        sum_sr_1 += 1
        sum_sr_2 += 1
        sum_sr_3 += 1
        sum_sr_4 += 1
        sum_sr_5 += 1
        sum_sr_6 += 1
        sum_sr_7 += 1
        sum_sr_8 += 1
        sum_sr_9 += 1
        sum_sr_10 += 1
    elif predictions[ns[len(ns)-2]] in labels:
        sum_sr_2 += 1
        sum_sr_3 += 1
        sum_sr_4 += 1
        sum_sr_5 += 1
        sum_sr_6 += 1
        sum_sr_7 += 1
        sum_sr_8 += 1
        sum_sr_9 += 1
        sum_sr_10 += 1
    elif predictions[ns[len(ns)-3]] in labels:
        sum_sr_3 += 1
        sum_sr_4 += 1
        sum_sr_5 += 1
        sum_sr_6 += 1
        sum_sr_7 += 1
        sum_sr_8 += 1
        sum_sr_9 += 1
        sum_sr_10 += 1
    elif predictions[ns[len(ns)-4]] in labels:
        sum_sr_4 += 1
        sum_sr_5 += 1
        sum_sr_6 += 1
        sum_sr_7 += 1
        sum_sr_8 += 1
        sum_sr_9 += 1
        sum_sr_10 += 1
    elif predictions[ns[len(ns)-5]] in labels:
        sum_sr_5 += 1
        sum_sr_6 += 1
        sum_sr_7 += 1
        sum_sr_8 += 1
        sum_sr_9 += 1
        sum_sr_10 += 1
    elif predictions[ns[len(ns)-6]] in labels:
        sum_sr_6 += 1
        sum_sr_7 += 1
        sum_sr_8 += 1
        sum_sr_9 += 1
        sum_sr_10 += 1
    elif predictions[ns[len(ns)-7]] in labels:
        sum_sr_7 += 1
        sum_sr_8 += 1
        sum_sr_9 += 1
        sum_sr_10 += 1
    elif predictions[ns[len(ns)-8]] in labels:
        sum_sr_8 += 1
        sum_sr_9 += 1
        sum_sr_10 += 1
    elif predictions[ns[len(ns)-9]] in labels:
        sum_sr_9 += 1
        sum_sr_10 += 1
    elif predictions[ns[len(ns)-10]] in labels:
        sum_sr_10 += 1




    # K = 1
    if predictions[ns[len(ns)-1]] in labels:
        correct_prediction += 1
        # Head, Mid, Tail
        if predictions[ns[len(ns)-1]] in group_head:
            correct_prediction_head += 1
        elif predictions[ns[len(ns)-1]] in group_mid:
            correct_prediction_mid += 1
        elif predictions[ns[len(ns)-1]] in group_tail:
            correct_prediction_tail += 1
        if predictions[ns[len(ns)-1]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-1]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-1]], 0) + 1
    if predictions[ns[len(ns)-1]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-1]]] = prediction_not_seen.get(predictions[ns[len(ns)-1]], 0) + 1
    # sum_precision_1 += (correct_prediction / min(1, len(labels)))
    sum_precision_1 += (correct_prediction / 1)
    sum_recall_1 += correct_prediction
    sum_precision_1_head += (correct_prediction_head / 1)
    sum_recall_1_head += correct_prediction_head
    sum_precision_1_mid += (correct_prediction_mid / 1)
    sum_recall_1_mid += correct_prediction_mid
    sum_precision_1_tail += (correct_prediction_tail / 1)
    sum_recall_1_tail += correct_prediction_tail
    prediction_all_check[predictions[ns[len(ns) - 1]]] = prediction_all_check.get(predictions[ns[len(ns) - 1]], 0) + 1

    # K = 2
    if predictions[ns[len(ns)-2]] in labels:
        correct_prediction += 1
                # Head, Mid, Tail
        if predictions[ns[len(ns)-2]] in group_head:
            correct_prediction_head += 1
        elif predictions[ns[len(ns)-2]] in group_mid:
            correct_prediction_mid += 1
        elif predictions[ns[len(ns)-2]] in group_tail:
            correct_prediction_tail += 1
        if predictions[ns[len(ns)-2]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-2]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-2]], 0) + 1
    if predictions[ns[len(ns)-2]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-2]]] = prediction_not_seen.get(predictions[ns[len(ns)-2]], 0) + 1
    prediction_all_check[predictions[ns[len(ns) - 2]]] = prediction_all_check.get(predictions[ns[len(ns) - 2]], 0) + 1
    # # sum_precision_2 += (correct_prediction / (min(2, len(labels))))
    # sum_precision_2 += (correct_prediction / 2)
    # sum_recall_2 += (correct_prediction / len(labels))
    # sum_precision_2_head += (correct_prediction_head / 2)
    # sum_recall_2_head += (correct_prediction_head / len(labels))
    # sum_precision_2_mid += (correct_prediction_mid / 2)
    # sum_recall_2_mid += (correct_prediction_mid / len(labels))
    # sum_precision_2_tail += (correct_prediction_tail / 2)
    # sum_recall_2_tail += (correct_prediction_tail / len(labels))

    # K = 3
    if predictions[ns[len(ns)-3]] in labels:
        correct_prediction += 1
                # Head, Mid, Tail
        if predictions[ns[len(ns)-3]] in group_head:
            correct_prediction_head += 1
        elif predictions[ns[len(ns)-3]] in group_mid:
            correct_prediction_mid += 1
        elif predictions[ns[len(ns)-3]] in group_tail:
            correct_prediction_tail += 1
        if predictions[ns[len(ns)-3]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-3]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-3]], 0) + 1
    if predictions[ns[len(ns)-3]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-3]]] = prediction_not_seen.get(predictions[ns[len(ns)-3]], 0) + 1
    prediction_all_check[predictions[ns[len(ns) - 3]]] = prediction_all_check.get(predictions[ns[len(ns) - 3]], 0) + 1
    # sum_precision_3 += (correct_prediction / min(3, len(labels)))
    sum_precision_3 += (correct_prediction / 3)
    sum_recall_3 += correct_prediction
    sum_precision_3_head += (correct_prediction_head / 3)
    sum_recall_3_head += correct_prediction_head
    sum_precision_3_mid += (correct_prediction_mid / 3)
    sum_recall_3_mid += correct_prediction_mid
    sum_precision_3_tail += (correct_prediction_tail / 3)
    sum_recall_3_tail += correct_prediction_tail

    # K = 4
    if predictions[ns[len(ns)-4]] in labels:
        correct_prediction += 1
                # Head, Mid, Tail
        if predictions[ns[len(ns)-4]] in group_head:
            correct_prediction_head += 1
        elif predictions[ns[len(ns)-4]] in group_mid:
            correct_prediction_mid += 1
        elif predictions[ns[len(ns)-4]] in group_tail:
            correct_prediction_tail += 1
        if predictions[ns[len(ns)-4]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-4]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-4]], 0) + 1
    if predictions[ns[len(ns)-4]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-4]]] = prediction_not_seen.get(predictions[ns[len(ns)-4]], 0) + 1
    prediction_all_check[predictions[ns[len(ns) - 4]]] = prediction_all_check.get(predictions[ns[len(ns) - 4]], 0) + 1
    # sum_precision_4 += (correct_prediction / min(4, len(labels)))
    # sum_precision_4 += (correct_prediction / 4)
    # sum_recall_4 += (correct_prediction / len(labels))
    # sum_precision_4_head += (correct_prediction_head / 4)
    # sum_recall_4_head += (correct_prediction_head / len(labels))
    # sum_precision_4_mid += (correct_prediction_mid / 4)
    # sum_recall_4_mid += (correct_prediction_mid / len(labels))
    # sum_precision_4_tail += (correct_prediction_tail / 4)
    # sum_recall_4_tail += (correct_prediction_tail / len(labels))
    # K = 5
    if predictions[ns[len(ns)-5]] in labels:
        correct_prediction += 1
                # Head, Mid, Tail
        if predictions[ns[len(ns)-5]] in group_head:
            correct_prediction_head += 1
        elif predictions[ns[len(ns)-5]] in group_mid:
            correct_prediction_mid += 1
        elif predictions[ns[len(ns)-5]] in group_tail:
            correct_prediction_tail += 1
        if predictions[ns[len(ns)-5]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-5]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-5]], 0) + 1
    if predictions[ns[len(ns)-5]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-5]]] = prediction_not_seen.get(predictions[ns[len(ns)-5]], 0) + 1

    prediction_all_check[predictions[ns[len(ns) - 5]]] = prediction_all_check.get(predictions[ns[len(ns) - 5]], 0) + 1
    # sum_precision_5 += (correct_prediction / min(5, len(labels)))
    sum_precision_5 += (correct_prediction / 5)
    sum_recall_5 += correct_prediction
    sum_precision_5_head += (correct_prediction_head / 5)
    sum_recall_5_head += correct_prediction_head
    sum_precision_5_mid += (correct_prediction_mid / 5)
    sum_recall_5_mid += correct_prediction_mid
    sum_precision_5_tail += (correct_prediction_tail / 5)
    sum_recall_5_tail += correct_prediction_tail


    # K = 6
    if predictions[ns[len(ns)-6]] in labels:
        correct_prediction += 1
                # Head, Mid, Tail
        if predictions[ns[len(ns)-6]] in group_head:
            correct_prediction_head += 1
        elif predictions[ns[len(ns)-6]] in group_mid:
            correct_prediction_mid += 1
        elif predictions[ns[len(ns)-6]] in group_tail:
            correct_prediction_tail += 1
        if predictions[ns[len(ns)-6]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-6]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-6]], 0) + 1
    if predictions[ns[len(ns)-6]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-6]]] = prediction_not_seen.get(predictions[ns[len(ns)-6]], 0) + 1
    # sum_precision_6 += (correct_prediction / min(6, len(labels)))

    prediction_all_check[predictions[ns[len(ns) - 6]]] = prediction_all_check.get(predictions[ns[len(ns) - 6]], 0) + 1
    # sum_precision_6 += (correct_prediction / 6)
    # sum_recall_6 += (correct_prediction / len(labels))
    # sum_precision_6_head += (correct_prediction_head / 6)
    # sum_recall_6_head += (correct_prediction_head / len(labels))
    # sum_precision_6_mid += (correct_prediction_mid / 6)
    # sum_recall_6_mid += (correct_prediction_mid / len(labels))
    # sum_precision_6_tail += (correct_prediction_tail / 6)
    # sum_recall_6_tail += (correct_prediction_tail / len(labels))

    # K = 7
    if predictions[ns[len(ns)-7]] in labels:
        correct_prediction += 1
                # Head, Mid, Tail
        if predictions[ns[len(ns)-7]] in group_head:
            correct_prediction_head += 1
        elif predictions[ns[len(ns)-7]] in group_mid:
            correct_prediction_mid += 1
        elif predictions[ns[len(ns)-7]] in group_tail:
            correct_prediction_tail += 1
        if predictions[ns[len(ns)-7]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-7]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-7]], 0) + 1
    if predictions[ns[len(ns)-7]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-7]]] = prediction_not_seen.get(predictions[ns[len(ns)-7]], 0) + 1
    # sum_precision_7 += (correct_prediction / min(7, len(labels)))
    sum_precision_7 += (correct_prediction / 7)

    prediction_all_check[predictions[ns[len(ns) - 7]]] = prediction_all_check.get(predictions[ns[len(ns) - 7]], 0) + 1
    sum_recall_7 += (correct_prediction / len(labels))


    # K = 8
    if predictions[ns[len(ns)-8]] in labels:
        correct_prediction += 1
                # Head, Mid, Tail
        if predictions[ns[len(ns)-8]] in group_head:
            correct_prediction_head += 1
        elif predictions[ns[len(ns)-8]] in group_mid:
            correct_prediction_mid += 1
        elif predictions[ns[len(ns)-8]] in group_tail:
            correct_prediction_tail += 1
        if predictions[ns[len(ns)-8]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-8]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-8]], 0) + 1
    if predictions[ns[len(ns)-8]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-8]]] = prediction_not_seen.get(predictions[ns[len(ns)-8]], 0) + 1
    # sum_precision_8 += (correct_prediction / min(8, len(labels)))
    sum_precision_8 += (correct_prediction / 8)

    prediction_all_check[predictions[ns[len(ns) - 8]]] = prediction_all_check.get(predictions[ns[len(ns) - 8]], 0) + 1
    sum_recall_8 += (correct_prediction / len(labels))


    # K = 9
    if predictions[ns[len(ns)-9]] in labels:
        correct_prediction += 1
                # Head, Mid, Tail
        if predictions[ns[len(ns)-9]] in group_head:
            correct_prediction_head += 1
        elif predictions[ns[len(ns)-9]] in group_mid:
            correct_prediction_mid += 1
        elif predictions[ns[len(ns)-9]] in group_tail:
            correct_prediction_tail += 1
        if predictions[ns[len(ns)-9]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-9]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-9]], 0) + 1
    if predictions[ns[len(ns)-9]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-9]]] = prediction_not_seen.get(predictions[ns[len(ns)-9]], 0) + 1
    # sum_precision_9 += (correct_prediction / min(9, len(labels)))
    sum_precision_9 += (correct_prediction / 9)

    prediction_all_check[predictions[ns[len(ns) - 9]]] = prediction_all_check.get(predictions[ns[len(ns) - 9]], 0) + 1
    sum_recall_9 += (correct_prediction / len(labels))


    # K = 10
    if predictions[ns[len(ns)-10]] in labels:
        correct_prediction += 1
                # Head, Mid, Tail
        if predictions[ns[len(ns)-10]] in group_head:
            correct_prediction_head += 1
        elif predictions[ns[len(ns)-10]] in group_mid:
            correct_prediction_mid += 1
        elif predictions[ns[len(ns)-10]] in group_tail:
            correct_prediction_tail += 1
        if predictions[ns[len(ns)-10]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-10]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-10]], 0) + 1
    if predictions[ns[len(ns)-10]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-10]]] = prediction_not_seen.get(predictions[ns[len(ns)-10]], 0) + 1
    # sum_precision_10 += (correct_prediction / min(10, len(labels)))
    sum_precision_10 += (correct_prediction / 10)

    prediction_all_check[predictions[ns[len(ns) - 10]]] = prediction_all_check.get(predictions[ns[len(ns) - 10]], 0) + 1
    sum_recall_10 += (correct_prediction / len(labels))
precision_1 = sum_precision_1 / num_test_data
precision_2 = sum_precision_2 / num_test_data
precision_3 = sum_precision_3 / num_test_data
precision_4 = sum_precision_4 / num_test_data
precision_5 = sum_precision_5 / num_test_data
recall_1 = sum_recall_1 / total_labels
recall_2 = sum_recall_2 / total_labels
recall_3 = sum_recall_3 / total_labels
recall_4 = sum_recall_4 / total_labels
recall_5 = sum_recall_5 / total_labels
sr_1 = sum_sr_1 / num_test_data
sr_2 = sum_sr_2 / num_test_data
sr_3 = sum_sr_3 / num_test_data
sr_4 = sum_sr_4 / num_test_data
sr_5 = sum_sr_5 / num_test_data
f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
# f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2)
f1_3 = 2 * precision_3 * recall_3 / (precision_3 + recall_3)
# f1_4 = 2 * precision_4 * recall_4 / (precision_4 + recall_4)
f1_5 = 2 * precision_5 * recall_5 / (precision_5 + recall_5)

#Head 
precision_1_head = sum_precision_1_head / num_test_data
precision_2_head = sum_precision_2_head / num_test_data
precision_3_head = sum_precision_3_head / num_test_data
precision_4_head = sum_precision_4_head / num_test_data
precision_5_head = sum_precision_5_head / num_test_data
recall_1_head = sum_recall_1_head / total_labels_head
recall_2_head = sum_recall_2_head / total_labels_head
recall_3_head = sum_recall_3_head / total_labels_head
recall_4_head = sum_recall_4_head / total_labels_head
recall_5_head = sum_recall_5_head / total_labels_head
sr_1_head = sum_sr_1_head / num_test_data
sr_2_head = sum_sr_2_head / num_test_data
sr_3_head = sum_sr_3_head / num_test_data
sr_4_head = sum_sr_4_head / num_test_data
sr_5_head = sum_sr_5_head / num_test_data
f1_1_head = 2 * precision_1_head * recall_1_head / (precision_1_head + recall_1_head)
# f1_2_head = 2 * precision_2_head * recall_2_head / (precision_2_head + recall_2_head)
f1_3_head = 2 * precision_3_head * recall_3_head / (precision_3_head + recall_3_head)
# f1_4_head = 2 * precision_4_head * recall_4_head / (precision_4_head + recall_4_head)
f1_5_head = 2 * precision_5_head * recall_5_head / (precision_5_head + recall_5_head)

#Mid
precision_1_mid = sum_precision_1_mid / num_test_data
precision_2_mid = sum_precision_2_mid / num_test_data
precision_3_mid = sum_precision_3_mid / num_test_data
precision_4_mid = sum_precision_4_mid / num_test_data
precision_5_mid = sum_precision_5_mid / num_test_data
recall_1_mid = sum_recall_1_mid / total_labels_mid
recall_2_mid = sum_recall_2_mid / total_labels_mid
recall_3_mid = sum_recall_3_mid / total_labels_mid
recall_4_mid = sum_recall_4_mid / total_labels_mid
recall_5_mid = sum_recall_5_mid / total_labels_mid
sr_1_mid = sum_sr_1_mid / num_test_data
sr_2_mid = sum_sr_2_mid / num_test_data
sr_3_mid = sum_sr_3_mid / num_test_data
sr_4_mid = sum_sr_4_mid / num_test_data
sr_5_mid = sum_sr_5_mid / num_test_data
f1_1_mid = 2 * precision_1_mid * recall_1_mid / (precision_1_mid + recall_1_mid)
# f1_2_mid = 2 * precision_2_mid * recall_2_mid / (precision_2_mid + recall_2_mid)
f1_3_mid = 2 * precision_3_mid * recall_3_mid / (precision_3_mid + recall_3_mid)
# f1_4_mid = 2 * precision_4_mid * recall_4_mid / (precision_4_mid + recall_4_mid)
f1_5_mid = 2 * precision_5_mid * recall_5_mid / (precision_5_mid + recall_5_mid)

#Tail
precision_1_tail = sum_precision_1_tail / num_test_data
precision_2_tail = sum_precision_2_tail / num_test_data
precision_3_tail = sum_precision_3_tail / num_test_data
precision_4_tail = sum_precision_4_tail / num_test_data
precision_5_tail = sum_precision_5_tail / num_test_data
recall_1_tail = sum_recall_1_tail / total_labels_tail
recall_2_tail = sum_recall_2_tail / total_labels_tail
recall_3_tail = sum_recall_3_tail / total_labels_tail
recall_4_tail = sum_recall_4_tail / total_labels_tail
recall_5_tail = sum_recall_5_tail / total_labels_tail
sr_1_tail = sum_sr_1_tail / num_test_data
sr_2_tail = sum_sr_2_tail / num_test_data
sr_3_tail = sum_sr_3_tail / num_test_data
sr_4_tail = sum_sr_4_tail / num_test_data
sr_5_tail = sum_sr_5_tail / num_test_data

if(precision_1_tail + recall_1_tail != 0):
    f1_1_tail = 2 * precision_1_tail * recall_1_tail / (precision_1_tail + recall_1_tail)
else:
    f1_1_tail = 0

# f1_2_tail = 2 * precision_2_tail * recall_2_tail / (precision_2_tail + recall_2_tail)
f1_3_tail = 2 * precision_3_tail * recall_3_tail / (precision_3_tail + recall_3_tail)
# f1_4_tail = 2 * precision_4_tail * recall_4_tail / (precision_4_tail + recall_4_tail)
f1_5_tail = 2 * precision_5_tail * recall_5_tail / (precision_5_tail + recall_5_tail)



precision_6 = sum_precision_6 / num_test_data
recall_6 = sum_recall_6 / num_test_data
sr_6 = sum_sr_6 / num_test_data
# f1_6 = 2 * precision_6 * recall_6 / (precision_6 + recall_6)


precision_7 = sum_precision_7 / num_test_data
recall_7 = sum_recall_7 / num_test_data
sr_7 = sum_sr_7 / num_test_data
# f1_7 = 2 * precision_7 * recall_7 / (precision_7 + recall_7)


precision_8 = sum_precision_8 / num_test_data
recall_8 = sum_recall_8 / num_test_data
sr_8 = sum_sr_8 / num_test_data
# f1_8 = 2 * precision_8 * recall_8 / (precision_8 + recall_8)


precision_9 = sum_precision_9 / num_test_data
recall_9 = sum_recall_9 / num_test_data
sr_9 = sum_sr_9 / num_test_data
# f1_9 = 2 * precision_9 * recall_9 / (precision_9 + recall_9)


precision_10 = sum_precision_10 / num_test_data
recall_10 = sum_recall_10 / num_test_data
sr_10 = sum_sr_10 / num_test_data
# f1_10 = 2 * precision_10 * recall_10 / (precision_10 + recall_10)

print("K = 1")
print("P@1 = " + precision_1.__str__())
print("R@1 = " + recall_1.__str__())
print("F@1 = " + f1_1.__str__())
print("SR@1 = " + sr_1.__str__())

print("P@1_head = " + precision_1_head.__str__())
print("R@1_head = " + recall_1_head.__str__())
print("F@1_head = " + f1_1_head.__str__())
print("SR@1_head = " + sr_1_head.__str__())

print("P@1_mid = " + precision_1_mid.__str__())
print("R@1_mid = " + recall_1_mid.__str__())
print("F@1_mid = " + f1_1_mid.__str__())
print("SR@1_mid = " + sr_1_mid.__str__())

print("P@1_tail = " + precision_1_tail.__str__())
print("R@1_tail = " + recall_1_tail.__str__())
print("F@1_tail = " + f1_1_tail.__str__())
print("SR@1_tail = " + sr_1_tail.__str__())


# print("K = 2")
# print("P@2 = " + precision_2.__str__())
# print("R@2 = " + recall_2.__str__())
# # print("F@2 = " + f1_2.__str__())
# print("SR@2 = " + sr_2.__str__())
# # Head
# print("K = 2 (Head)")
# print("P@2_head = " + precision_2_head.__str__())
# print("R@2_head = " + recall_2_head.__str__())
# # print("F@2_head = " + f1_2_head.__str__())
# print("SR@2_head = " + sr_2_head.__str__())

# # Mid
# print("K = 2 (Mid)")
# print("P@2_mid = " + precision_2_mid.__str__())
# print("R@2_mid = " + recall_2_mid.__str__())
# # print("F@2_mid = " + f1_2_mid.__str__())
# print("SR@2_mid = " + sr_2_mid.__str__())

# # Tail
# print("K = 2 (Tail)")
# print("P@2_tail = " + precision_2_tail.__str__())
# print("R@2_tail = " + recall_2_tail.__str__())
# # print("F@2_tail = " + f1_2_tail.__str__())
# print("SR@2_tail = " + sr_2_tail.__str__())


print("K = 3")
print("P@3 = " + precision_3.__str__())
print("R@3 = " + recall_3.__str__())
print("F@3 = " + f1_3.__str__())
print("SR@3 = " + sr_3.__str__())
print("TOTAL LABELS: " + total_labels.__str__())

# Head
print("K = 3 (Head)")
print("P@3_head = " + precision_3_head.__str__())
print("R@3_head = " + recall_3_head.__str__())
print("F@3_head = " + f1_3_head.__str__())
print("SR@3_head = " + sr_3_head.__str__())

# Mid
print("K = 3 (Mid)")
print("P@3_mid = " + precision_3_mid.__str__())
print("R@3_mid = " + recall_3_mid.__str__())
print("F@3_mid = " + f1_3_mid.__str__())
print("SR@3_mid = " + sr_3_mid.__str__())

# Tail
print("K = 3 (Tail)")
print("P@3_tail = " + precision_3_tail.__str__())
print("R@3_tail = " + recall_3_tail.__str__())
print("F@3_tail = " + f1_3_tail.__str__())
print("SR@3_tail = " + sr_3_tail.__str__())


# print("K = 4")
# print("P@4 = " + precision_4.__str__())
# print("R@4 = " + recall_4.__str__())
# print("F@4 = " + f1_4.__str__())
# print("SR@4 = " + sr_4.__str__())
# print("TOTAL DATA: " + num_test_data.__str__())

print("K = 5")
print("P@5 = " + precision_5.__str__())
print("R@5 = " + recall_5.__str__())
print("F@5 = " + f1_5.__str__())
print("SR@5 = " + sr_5.__str__())
print("TOTAL LABELS: " + total_labels.__str__())

# Head
print("K = 5 (Head)")
print("P@5_head = " + precision_5_head.__str__())
print("R@5_head = " + recall_5_head.__str__())
print("F@5_head = " + f1_5_head.__str__())
print("SR@5_head = " + sr_5_head.__str__())

# Mid
print("K = 5 (Mid)")
print("P@5_mid = " + precision_5_mid.__str__())
print("R@5_mid = " + recall_5_mid.__str__())
print("F@5_mid = " + f1_5_mid.__str__())
print("SR@5_mid = " + sr_5_mid.__str__())

# Tail
print("K = 5 (Tail)")
print("P@5_tail = " + precision_5_tail.__str__())
print("R@5_tail = " + recall_5_tail.__str__())
print("F@5_tail = " + f1_5_tail.__str__())
print("SR@5_tail = " + sr_5_tail.__str__())


# print("K = 6")
# print("P@6 = " + precision_6.__str__())
# print("R@6 = " + recall_6.__str__())
# # print("F@6 = " + f1_6.__str__())
# print("SR@6 = " + sr_6.__str__())

# print("K = 7")
# print("P@7 = " + precision_7.__str__())
# print("R@7 = " + recall_7.__str__())
# print("F@7 = " + f1_7.__str__())
# print("SR@7 = " + sr_7.__str__())

# print("K = 8")
# print("P@8 = " + precision_8.__str__())
# print("R@8 = " + recall_8.__str__())
# print("F@8 = " + f1_8.__str__())
# print("SR@8 = " + sr_8.__str__())

# print("K = 9")
# print("P@9 = " + precision_9.__str__())
# print("R@9 = " + recall_9.__str__())
# print("F@9 = " + f1_9.__str__())
# print("SR@9 = " + sr_9.__str__())

# print("K = 10")
# print("P@10 = " + precision_10.__str__())
# print("R@10 = " + recall_10.__str__())
# print("F@10 = " + f1_10.__str__())
# print("SR@10 = " + sr_10.__str__())

# print(prediction_not_seen)
print("How many unseen labels:")
print(len(prediction_not_seen))
print("How many unseen labels usage")
sum = 0
print(prediction_not_seen)
for key, items in prediction_not_seen.items():
    sum += items
print(sum)
print()
print()
# print(prediction_not_seen_correct)
print("How many unseen labels correct:")
print(len(prediction_not_seen_correct))
print("How many unseen labels used correctly")
sum = 0
for key, items in prediction_not_seen_correct.items():
    sum += items
print(sum)


print("How many unique labels:")
print(len(prediction_all_check))
print("How many labels usage")
sum = 0
# print(prediction_not_seen)
for key, items in prediction_all_check.items():
    sum += items
print(sum)
