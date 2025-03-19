import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
import sklearn
import pickle
import os

DATASET_PATH = "dataset/without_lemma_dataset_train_base.csv"

# According to the usual division, we divide the dataset into 0.75:0.25 between the training and test data
# the splitted result will be saved in the dataset/splitted folder into 4 different files:
# splitted_train_x = training data (description/reference/etc.)
# splitted_train_y = label for training data
# splitted_test_x = test data
# splitted_test_y = label for test data


def updated_readme_description_save_splitted_dataset_as_numpy():
    TRAIN_PATH = "dataset/without_lemma_dataset_train_base.csv"
    TEST_PATH = "dataset/without_lemma_dataset_test_base.csv"
    # description_fields = ["cve_id", "merged"]
    description_fields = ["id", "readme", "description"]
    # Initiate the dataframe containing the CVE ID and its description
    # Change the "merged" field in the description_fields variable to use other text feature such as reference

    # Process the training dataset
    print("part-0")
    df = pd.read_csv(TRAIN_PATH, usecols=description_fields, delimiter=",", na_filter=False, low_memory=False)
    df["readme"] = df["readme"] +" "+ df["description"]
    df = df.drop('description', axis=1)
    print("part-1")
    print("part-2")

    pd_labels = pd.read_csv(TRAIN_PATH, delimiter=",", na_filter=False, low_memory=False)
    pd_labels.drop(["id", "readme", "description", "repo"], axis=1, inplace=True)

    print("part-3")
    # Convert to numpy for splitting
    train = df.to_numpy()
    label_train = pd_labels.to_numpy()
    # Splitting using skmultilearn iterative train test split
    print("part-3.5")
    df_test = pd.read_csv(TEST_PATH, usecols=description_fields, delimiter=",", na_filter=False, low_memory=False)
    df_test["readme"] = df_test["readme"] +" "+ df_test["description"]
    df_test = df_test.drop('description', axis=1)
    print("part-4")
    pd_labels_test = pd.read_csv(TEST_PATH, delimiter=",", na_filter=False, low_memory=False)
    pd_labels_test.drop(["id", "readme", "description", "repo"], axis=1, inplace=True)
    test = df_test.to_numpy()
    label_test = pd_labels_test.to_numpy()

    os.makedirs("dataset/splitted", exist_ok=True)
    # Save the splitted data to files
    np.save("dataset/splitted/splitted_train_x.npy", train, allow_pickle=True)
    np.save("dataset/splitted/splitted_train_y.npy", label_train, allow_pickle=True)
    np.save("dataset/splitted/splitted_test_x.npy", test, allow_pickle=True)
    np.save("dataset/splitted/splitted_test_y.npy", label_test, allow_pickle=True)



def updated_readme_description_prepare_omikuji_dataset():
    # Load the splitted dataset files
    train = np.load("dataset/splitted/splitted_train_x.npy", allow_pickle=True)
    label_train = np.load("dataset/splitted/splitted_train_y.npy", allow_pickle=True)
    test = np.load("dataset/splitted/splitted_test_x.npy", allow_pickle=True)
    label_test = np.load("dataset/splitted/splitted_test_y.npy", allow_pickle=True)
    train_corpus = train[:, 1].tolist()
    test_corpus = test[:, 1].tolist()
    cols = list(pd.read_csv(DATASET_PATH, nrows=1, delimiter=",", na_filter=False, low_memory=False))
    # label_columns = [i for i in cols if i not in ["id", "text"]]
    label_columns = [i for i in cols if i not in ["id", "repo", "readme", "description"]]
    num_labels = len(label_columns)
    print("VECTORIZER")
    vectorizer = TfidfVectorizer().fit(train_corpus)

    idx_zero_train = np.argwhere(np.all(label_train[..., :] == 0, axis=0))
    idx_zero_test = np.argwhere(np.all(label_test[..., :] == 0, axis=0))

    train_X = vectorizer.transform(train_corpus)
    # train_Y = np.delete(label_train, idx_zero_train, axis=1)
    train_Y = label_train
    test_X = vectorizer.transform(test_corpus)
    # test_Y = np.delete(label_test, idx_zero_test, axis=1)
    test_Y = label_test

    num_features = len(vectorizer.get_feature_names_out())
    num_row_train = train_X.shape[0]
    num_row_test = test_X.shape[0]
    train_file_header = num_row_train.__str__() + " " + num_features.__str__() + " " + (num_labels).__str__()
    test_file_header = num_row_test.__str__() + " " + num_features.__str__() + " " + (num_labels).__str__()


    os.makedirs("dataset/omikuji/", exist_ok=True)
    # Dump the standard svmlight file
    dump_svmlight_file(train_X, train_Y, "dataset/omikuji/train.txt", multilabel=True)
    dump_svmlight_file(test_X, test_Y, "dataset/omikuji/test.txt", multilabel=True)
    # Prepend the header to the svmlight file


    with open("dataset/omikuji/train.txt", 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(train_file_header.rstrip('\r\n') + '\n' + content)
        f.close()

    with open("dataset/omikuji/test.txt", 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(test_file_header.rstrip('\r\n') + '\n' + content)
        f.close()

# the test and train data are the same with omikuji
# however, you need to create the train/test_labels.txt and train/test_texts.txt
# with each row contains the text and labels for the train/test data
def prepare_lightxml_dataset(TRAIN_CSV_PATH, TEST_CSV_PATH, LABELS_PATH):
    # Load the splitted dataset files
    train = np.load("dataset/splitted/splitted_train_x.npy", allow_pickle=True)
    label_train = np.load("dataset/splitted/splitted_train_y.npy", allow_pickle=True)
    test = np.load("dataset/splitted/splitted_test_x.npy", allow_pickle=True)
    label_test = np.load("dataset/splitted/splitted_test_y.npy", allow_pickle=True)

    train_corpus = train[:, 1].tolist()
    test_corpus = test[:, 1].tolist()
    cols = list(pd.read_csv(TRAIN_CSV_PATH, nrows=1))
    label_columns = [i for i in cols if i not in ["id", "readme", "repo", "description"]]
    num_labels = len(label_columns)

    vectorizer = TfidfVectorizer().fit(train_corpus)

    idx_zero_train = np.argwhere(np.all(label_train[..., :] == 0, axis=0))
    idx_zero_test = np.argwhere(np.all(label_test[..., :] == 0, axis=0))

    train_X = vectorizer.transform(train_corpus)
    # train_Y = np.delete(label_train, idx_zero_train, axis=1)
    train_Y = label_train
    test_X = vectorizer.transform(test_corpus)
    # test_Y = np.delete(label_test, idx_zero_test, axis=1)
    test_Y = label_test

    num_features = len(vectorizer.get_feature_names_out())
    num_row_train = train_X.shape[0]
    num_row_test = test_X.shape[0]

    os.makedirs("dataset/lightxml/", exist_ok=True)
    # Dump the standard svmlight file
    dump_svmlight_file(train_X, train_Y, "dataset/lightxml/train.txt", multilabel=True)
    dump_svmlight_file(test_X, test_Y, "dataset/lightxml/test.txt", multilabel=True)

    train_text = []
    train_label = []
    test_text = []
    test_label = []

    topic_labels = pd.read_csv(LABELS_PATH)
    print("READ LABEL")
    print(topic_labels)

    train_data = pd.read_csv(TRAIN_CSV_PATH)
    train_data["readme"] = train_data["readme"] + " " + train_data["description"]

    # process the label and text here
    for index, row in train_data.iterrows():
        train_text.append(row.readme.lstrip().rstrip())
        # for label below
        label = topic_labels[topic_labels["id"] == row.id]
        label_unsplit = label.labels.values[0]
        label_array = label_unsplit.split(",")
        label_string = ""
        for label in label_array:
            label_string = label_string + label + " "
        label_string = label_string.rstrip()
        print(index)
        print(label_string)
        train_label.append(label_string)

    test_data = pd.read_csv(TEST_CSV_PATH)
    test_data["readme"] = test_data["readme"] + " " + test_data["description"]
    for index, row in test_data.iterrows():
        test_text.append(row.readme.lstrip().rstrip())
        # for label below
        label = topic_labels[topic_labels["id"] == row.id]
        label_unsplit = label.labels.values[0]
        label_array = label_unsplit.split(",")
        label_string = ""
        for label in label_array:
            label_string = label_string + label + " "
        label_string = label_string.rstrip()
        print(index)
        print(label_string)
        test_label.append(label_string)


    with open("dataset/lightxml/train_texts.txt", "w", encoding="utf-8") as wr:
        for line in train_text:
            wr.write(line + "\n")

    with open("dataset/lightxml/train_labels.txt", "w", encoding="utf-8") as wr:
        for line in train_label:
            wr.write(line + "\n")

    with open("dataset/lightxml/test_texts.txt", "w", encoding="utf-8") as wr:
        for line in test_text:
            wr.write(line + "\n")

    with open("dataset/lightxml/test_labels.txt", "w", encoding="utf-8") as wr:
        for line in test_label:
            wr.write(line + "\n")




