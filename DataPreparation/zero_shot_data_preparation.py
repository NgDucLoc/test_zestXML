import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
import json
import sklearn
import pickle
import os
import pandas as pd

DATASET_PATH = "dataset/zero_shot_dataset_combined/zero_shot_train_cleaned.csv"
NON_LABEL_COLUMNS = ["id", "repo", "readme", "description"]
BASE_PATH = "dataset/"

XF_ROW = 0
YF_ROW = 0


def get_row_Xf():
    num_lines = sum(1 for line in open('dataset/zero_shot_dataset_combined/zestxml/Xf.txt'))
    return str(num_lines)


def get_row_Yf():
    num_lines = sum(1 for line in open('dataset/zero_shot_dataset_combined/zestxml/Yf.txt'))
    return str(num_lines)


# Split the data for zero shot training
# The dataset is split chronologically
# 75% training, 25% testing just like Chen et al. paper
# The idea is that the test data is newer and will contain new libraries that are not seen in the training data
def zero_shot_data_splitting():
    original_merged_cleaned_data_path = "dataset/without_lemma_dataset_train.csv"
    df = pd.read_csv(original_merged_cleaned_data_path, delimiter=",", na_filter=False, low_memory=False)
    print("finish reading")
    os.makedirs("dataset/zero_shot_dataset_combined/", exist_ok=True)
    mask = np.array([True] * int(float(len(df)) * 0.90) + [False] * int(float(len(df)) * 0.10) + [False])
    train = df[mask]
    print("masking")
    train.to_csv("dataset/zero_shot_dataset_combined/zero_shot_train_cleaned.csv", index=False)
    print("training data")

    test = pd.read_csv("dataset/without_lemma_dataset_test.csv", delimiter=",", na_filter=False, low_memory=False,
                       dtype=str)
    print("testing data")
    val = df[~mask]
    test.to_csv("dataset/zero_shot_dataset_combined/zero_shot_test_cleaned.csv", index=False)
    val.to_csv("dataset/zero_shot_dataset_combined/zero_shot_val_cleaned.csv", index=False)


# this function is used to save the splitted dataset as numpy file
# use this function if the csv files are already splitted into the test and train dataset
def save_splitted_zero_shot_dataset_combined_validation_as_numpy():
    TRAIN_PATH = "dataset/zero_shot_dataset_combined/zero_shot_train_cleaned.csv"
    TEST_PATH = "dataset/zero_shot_dataset_combined/zero_shot_test_cleaned.csv"
    VAL_PATH = "dataset/zero_shot_dataset_combined/zero_shot_val_cleaned.csv"
    description_fields = ["id", "readme"]
    # Initiate the dataframe containing the CVE ID and its description
    # Change the "text" field in the description_fields variable to use other text feature such as reference

    # Process the training dataset

    df = pd.read_csv(TRAIN_PATH, usecols=description_fields, delimiter=",", na_filter=False, low_memory=False)
    print("read_train")
    df.readme = df.readme.astype(str)
    # Read column names from file
    # cols = list(pd.read_csv(TRAIN_PATH, nrows=1))
    # Initiate the dataframe containing the labels for each CVE

    # pd_labels = pd.read_csv(TRAIN_PATH,
    #                         usecols=[i for i in cols if i not in NON_LABEL_COLUMNS])
    pd_labels = pd.read_csv(TRAIN_PATH, delimiter=",", na_filter=False, low_memory=False)
    pd_labels.drop(NON_LABEL_COLUMNS, axis=1, inplace=True)
    # Initiate a list which contain the list of labels considered in te dataset
    # list_labels = [i for i in cols if i not in NON_LABEL_COLUMNS]
    print("read another train")
    # Convert to numpy for splitting
    train = df.to_numpy()
    label_train = pd_labels.to_numpy()
    print("to nmpy")
    df_test = pd.read_csv(TEST_PATH, usecols=description_fields, delimiter=",", na_filter=False, low_memory=False)
    df_test.readme = df_test.readme.astype(str)
    print("read df")
    pd_labels_test = pd.read_csv(TEST_PATH, delimiter=",", na_filter=False, low_memory=False)
    pd_labels_test.drop(NON_LABEL_COLUMNS, axis=1, inplace=True)
    print("get_label")
    test = df_test.to_numpy()
    label_test = pd_labels_test.to_numpy()
    print("get val")
    df_val = pd.read_csv(VAL_PATH, usecols=description_fields, delimiter=",", na_filter=False, low_memory=False)
    df_val.readme = df_val.readme.astype(str)
    print("get another val")
    pd_labels_val = pd.read_csv(VAL_PATH, delimiter=",", na_filter=False, low_memory=False)
    pd_labels_val.drop(NON_LABEL_COLUMNS, axis=1, inplace=True)
    val = df_val.to_numpy()
    label_val = pd_labels_val.to_numpy()

    # Save the splitted data to files
    os.makedirs("dataset/zero_shot_dataset_combined/splitted_val", exist_ok=True)
    np.save("dataset/zero_shot_dataset_combined/splitted_val/splitted_train_x.npy", train, allow_pickle=True)
    np.save("dataset/zero_shot_dataset_combined/splitted_val/splitted_train_y.npy", label_train, allow_pickle=True)
    np.save("dataset/zero_shot_dataset_combined/splitted_val/splitted_test_x.npy", test, allow_pickle=True)
    np.save("dataset/zero_shot_dataset_combined/splitted_val/splitted_test_y.npy", label_test, allow_pickle=True)
    np.save("dataset/zero_shot_dataset_combined/splitted_val/splitted_val_x.npy", val, allow_pickle=True)
    np.save("dataset/zero_shot_dataset_combined/splitted_val/splitted_val_y.npy", label_val, allow_pickle=True)




# THERE ARE SO MANY THINGS TO PREPARE FOR ZESTXML:
# DONE Xf.txt: all features used in tf-idf representation of documents ((trn/tst/val)_X_Xf), ith line denotes ith feature in the tf-idf representation. In particular, for datasets used in the paper, it's the stemmed bigram and unigram features of documents but you can choose to have any set of features depending on your application.
# DONE Yf.txt: similar to Xf.txt it represents features of all labels. In addition to unigrams and bigrams, we also add a unique feature specific to each label (represented by __label__<i>__<label-i-text>, this feature will only be present in ith label's features), this allows the model to have label specific parameters and helps it to do well on many-shot labels. Features with __parent__ in them are only specific to the GZ-EURLex-4.3K dataset because raw labels in this dataset have some additional information about parent concepts of each label, you can safely choose to ignore these features for any other/new dataset.
# DONE (trn/tst/val)_X_Xf.txt: sparse matrix (documents x document-features) representing tf-idf feature matrix of (trn/tst/val) input documents.
# DONE Y_Yf.txt: similar to (trn/tst/val)_X_Xf.txt but for labels, this is the sparse matrix (labels x label-features) representing tf-idf feature matrix of labels.
# trn_Y_Yf.txt: similar to Y_Yf.txt but contains features for only the seen labels (can be interpreted as Y_Yf[seen-labels])
# DONE (trn/tst/val)_X_Y.txt: sparse matrix (documents x labels) representing (trn/tst/val) document-label relevance matrix.


# helper function for trn_Y_Yf
# get the list of seen labels from a csv file
TRAINING_DATA_PATH = "dataset/zero_shot_dataset_combined/zero_shot_train_cleaned.csv"

def get_list_labels(csv_file_path):
    seen_label = []

    COLUMNS = list(
        pd.read_csv(TRAINING_DATA_PATH, nrows=1, delimiter=",", na_filter=False, low_memory=False, dtype=str))
    # COLUMNS = "a"
    LABEL_COLUMNS = [i for i in COLUMNS if i not in ["id", "readme", "repo", "description"]]

    df = pd.read_csv(csv_file_path, usecols=LABEL_COLUMNS, delimiter=",", na_filter=False, low_memory=False)
    for label in LABEL_COLUMNS:
        sum = df[label].sum()
        if sum > 0:
            seen_label.append(label)
    return seen_label


# should be similar with the regular Y_Yf
# just need to find out which labels are seen in the training data
import regex as re


def prepare_zest_trn_Y_Yf(vectorizer):
    list_labels = get_list_labels(TRAINING_DATA_PATH)
    # add the unique label features
    for i in range(0, len(list_labels)):
        label = list_labels[i]
        s = re.sub(r"[^\w\s]", '_', label)
        # formatted = "__label__" + i.__str__() + "__" + label.replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_").replace(";", "_")
        formatted = "__label__" + i.__str__() + "__" + s.replace(" ", "_")
        list_labels[i] = list_labels[i] + " " + formatted

    with open("dataset/zero_shot_dataset_combined/zestxml/trn_Y_Yf.txt", "w", encoding="utf-8") as wr:
        # header is number of labels SPACE number of features (i.e., numrows of Yf.txt)
        wr.write(len(list_labels).__str__() + " " + YF_ROW + "\n")
        for label in list_labels:
            sparse_mat = vectorizer.transform([label])
            value = sparse_mat.data
            indices = sparse_mat.indices
            sorted_value = [x for _, x in sorted(zip(indices, value))]
            sorted_indices = sorted(indices)
            # printing the tfidf values
            to_print = ""
            for i in range(0, len(sorted_value)):
                to_print = to_print + sorted_indices[i].__str__() + ":" + sorted_value[i].__str__() + " "
            to_print = to_print[:-1] + "\n"
            wr.write(to_print)


# make use of the list of labels and the vectorizer created during the Yf.txt creation
# potentially buggy as we did not consider the __label__ features
# BUG IS FIXED
def prepare_zest_Y_Yf(label_column, vectorizer):
    with open("dataset/zero_shot_dataset_combined/zestxml/Y_Yf.txt", "w", encoding="utf-8") as wr:
        # header is number of labels SPACE number of features (i.e., numrows of Yf.txt)
        wr.write("37161 " + YF_ROW + "\n")
        for label in label_column:
            sparse_mat = vectorizer.transform([label])
            value = sparse_mat.data
            indices = sparse_mat.indices
            sorted_value = [x for _, x in sorted(zip(indices, value))]
            sorted_indices = sorted(indices)
            # printing the tfidf values
            to_print = ""
            for i in range(0, len(sorted_value)):
                to_print = to_print + sorted_indices[i].__str__() + ":" + sorted_value[i].__str__() + " "
            to_print = to_print[:-1] + "\n"
            wr.write(to_print)


# process from the svmlight file
# generates a total of 6 files, which are the three X_Xf files
# and the three X_Y files
def prepare_zest_X_Xf_and_X_Y():
    with open("dataset/zero_shot_dataset_combined/zestxml/trn_svmlight.txt", "r", encoding="utf-8") as re:
        lines = re.read().splitlines()
        num_rows = len(lines)
        xf_wr = open("dataset/zero_shot_dataset_combined/zestxml/trn_X_Xf.txt", "w", encoding="utf-8")
        xy_wr = open("dataset/zero_shot_dataset_combined/zestxml/trn_X_Y.txt", "w", encoding="utf-8")
        # write the header: num_rows num_cols
        # num_cols is taken from the Xf.txt and from the number of labels in the dataset respectively
        xf_wr.write(num_rows.__str__() + " " + XF_ROW + "\n")
        xy_wr.write(num_rows.__str__() + " 37161\n")
        for line in lines:
            line = line.strip()
            # split into 2, the [0] is labels, [1] is TfIdf features
            split = line.split(" ", 1)
            xf_wr.write(split[1] + "\n")
            # for the labels, split further based on comma
            label_text = ""
            for label in split[0].split(","):
                label_text = label_text + label + ":1.00000 "
            label_text = label_text[:-1] + "\n"
            xy_wr.write(label_text)
        xf_wr.close()
        xy_wr.close()
        re.close()

    with open("dataset/zero_shot_dataset_combined/zestxml/tst_svmlight.txt", "r", encoding="utf-8") as re:
        lines = re.read().splitlines()
        num_rows = len(lines)
        xf_wr = open("dataset/zero_shot_dataset_combined/zestxml/tst_X_Xf.txt", "w", encoding="utf-8")
        xy_wr = open("dataset/zero_shot_dataset_combined/zestxml/tst_X_Y.txt", "w", encoding="utf-8")
        # write the header: num_rows num_cols
        # num_cols is taken from the Xf.txt and from the number of labels in the dataset respectively
        xf_wr.write(num_rows.__str__() + " " + XF_ROW + "\n")
        xy_wr.write(num_rows.__str__() + " 37161\n")
        for line in lines:
            line = line.strip()
            # split into 2, the [0] is labels, [1] is TfIdf features
            split = line.split(" ", 1)
            xf_wr.write(split[1] + "\n")
            # for the labels, split further based on comma
            label_text = ""
            for label in split[0].split(","):
                label_text = label_text + label + ":1.00000 "
            label_text = label_text[:-1] + "\n"
            xy_wr.write(label_text)
        xf_wr.close()
        xy_wr.close()
        re.close()

    with open("dataset/zero_shot_dataset_combined/zestxml/val_svmlight.txt", "r", encoding="utf-8") as re:
        lines = re.read().splitlines()
        num_rows = len(lines)
        xf_wr = open("dataset/zero_shot_dataset_combined/zestxml/val_X_Xf.txt", "w", encoding="utf-8")
        xy_wr = open("dataset/zero_shot_dataset_combined/zestxml/val_X_Y.txt", "w", encoding="utf-8")
        # write the header: num_rows num_cols
        # num_cols is taken from the Xf.txt and from the number of labels in the dataset respectively
        xf_wr.write(num_rows.__str__() + " " + XF_ROW + "\n")
        xy_wr.write(num_rows.__str__() + " 37161\n")
        for line in lines:
            line = line.strip()
            # split into 2, the [0] is labels, [1] is TfIdf features
            split = line.split(" ", 1)
            xf_wr.write(split[1] + "\n")
            # for the labels, split further based on comma
            label_text = ""
            for label in split[0].split(","):
                label_text = label_text + label + ":1.00000 "
            label_text = label_text[:-1] + "\n"
            xy_wr.write(label_text)
        xf_wr.close()
        xy_wr.close()
        re.close()


# Possibly for this one is similar to SVMLight format without the labels at the beginning
# Use the Vectorizer created during the Xf.txt creation
# this function will generate the svmlight first
# which will then be processed into X_Xf.txt and trn/tst/val_X_Y.txt
def prepare_zest_svmlight(vectorizer):
    # Load the splitted dataset files
    train = np.load("dataset/zero_shot_dataset_combined/splitted_val/splitted_train_x.npy", allow_pickle=True)
    label_train = np.load("dataset/zero_shot_dataset_combined/splitted_val/splitted_train_y.npy", allow_pickle=True)
    test = np.load("dataset/zero_shot_dataset_combined/splitted_val/splitted_test_x.npy", allow_pickle=True)
    label_test = np.load("dataset/zero_shot_dataset_combined/splitted_val/splitted_test_y.npy", allow_pickle=True)
    val = np.load("dataset/zero_shot_dataset_combined/splitted_val/splitted_val_x.npy", allow_pickle=True)
    label_val = np.load("dataset/zero_shot_dataset_combined/splitted_val/splitted_val_y.npy", allow_pickle=True)
    train_corpus = train[:, 1].tolist()
    test_corpus = test[:, 1].tolist()
    val_corpus = val[:, 1].tolist()
    cols = list(pd.read_csv(DATASET_PATH, nrows=1))
    label_columns = [i for i in cols if i not in ["id", "readme", "repo", "description"]]

    vectorizer = vectorizer

    train_X = vectorizer.transform(train_corpus)
    train_Y = label_train
    test_X = vectorizer.transform(test_corpus)
    test_Y = label_test
    val_X = vectorizer.transform(val_corpus)
    val_Y = label_val

    # print(train_Y)
    # # Chuyển train_Y thành một pandas Series nếu chưa phải là Series
    # train_Y_series = pd.Series(train_Y)

    # # Kiểm tra giá trị null hoặc trống
    # if train_Y_series.isnull().any() or (train_Y_series == "").any() or (train_Y_series == []).any():
    #     print("train_Y có giá trị thiếu hoặc trống")
    # else:
    #     print("Không có giá trị thiếu hoặc trống trong train_Y")
    # print(train_Y[420:425])
    # Redirect output vào less
    # with open('output.txt', 'w') as f:
    #     np.savetxt(f, train_Y[420:425], fmt='%d')
    # Kiểm tra nếu có giá trị NaN hoặc None trong y_train
    if np.any(np.isnan(train_Y)) or np.any([y is None for y in train_Y]):
        print("Có giá trị NaN hoặc None trong y_train")
        # Loại bỏ hoặc thay thế giá trị không hợp lệ


    # Dump the standard svmlight file
    dump_svmlight_file(train_X, train_Y, "dataset/zero_shot_dataset_combined/zestxml/trn_svmlight.txt", multilabel=True)
    dump_svmlight_file(test_X, test_Y, "dataset/zero_shot_dataset_combined/zestxml/tst_svmlight.txt", multilabel=True)
    dump_svmlight_file(val_X, val_Y, "dataset/zero_shot_dataset_combined/zestxml/val_svmlight.txt", multilabel=True)
    #


# Prepare the Xf.txt, which contains all features used in tf-idf representation of documents
# Therefore I assume it would be
# 1. Create TfIdfVectorizer using all the text dataset
# 2. The TfIdfVectorizer uses Unigram and Bigram
# 3. Then, get the vocabulary dictionary (i.e., TfIdfVectorizer.vocabulary
# return the TfIdfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def prepare_zestxml_Xf():
    df = pd.read_csv("dataset/without_lemma_dataset_train.csv", usecols=["readme", "description"],
                     delimiter=",", na_filter=False, low_memory=False)
    df["readme"] = df["readme"] + " " + df["description"]
    df = df.drop('description', axis=1)
    print("readdattaset")
    text_corpus = df["readme"].values  # .astype("U")
    print("to_corpus")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(text_corpus)
    print("vectorizer")
    os.makedirs("dataset/zero_shot_dataset_combined/zestxml/", exist_ok=True)
    with open("dataset/zero_shot_dataset_combined/zestxml/Xf.txt", "w", encoding="utf-8") as wr:
        for key in sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get):
            wr.write(key + "\n")
    print("write_x")
    global XF_ROW
    XF_ROW = get_row_Xf()
    return vectorizer


# Simply the list of labels, in the form of unigram, bigram, and unique __label__ format
def prepare_zestxml_Yf():
    print("read label")
    cols = list(
        pd.read_csv("dataset/dataset_without_lemma.csv", nrows=1, delimiter=",", na_filter=False,
                    low_memory=False))
    label_columns = [i for i in cols if i not in NON_LABEL_COLUMNS]
    label_labels = []
    for i, label in enumerate(label_columns):
        import regex as re
        s = re.sub(r"[^\w\s]", '_', label)
        # formatted = "__label__" + i.__str__() + "__" + label.replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_").replace(";", "_")
        formatted = "__label__" + i.__str__() + "__" + s.replace(" ", "_")
        label_labels.append(formatted)
    print("start vectorizer")
    # print(label_labels)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit((label_columns + label_labels))
    print("write file yf")
    with open("dataset/zero_shot_dataset_combined/zestxml/Yf.txt", "w", encoding="utf-8") as wr:
        for key in sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get):
            wr.write(key + "\n")
        # for label in label_labels:
        #     wr.write(label + "\n")
    for i in range(0, len(label_columns)):
        label_columns[i] = label_columns[i] + " " + label_labels[i]
    global YF_ROW
    YF_ROW = get_row_Yf()
    return label_columns, vectorizer


def prepare_zestxml_dataset():
    tfidf_vectorizer = prepare_zestxml_Xf()
    label_column, label_vectorizer = prepare_zestxml_Yf()
    prepare_zest_svmlight(tfidf_vectorizer)
    prepare_zest_X_Xf_and_X_Y()
    prepare_zest_Y_Yf(label_column, label_vectorizer)
    prepare_zest_trn_Y_Yf(label_vectorizer)


if __name__ == "__main__":
    print("Data splitting start")
    zero_shot_data_splitting()
    print("Data splitting finished")
    save_splitted_zero_shot_dataset_combined_validation_as_numpy()
    print("Transform to numpy finished")
    prepare_zestxml_dataset()
    print("ZestXML data preparation finished")