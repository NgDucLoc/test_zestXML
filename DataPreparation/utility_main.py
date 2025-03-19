from data_preparation import *
from zero_shot_data_preparation import *



# #prepare bonsai, parabel, and omikuji data
# print("Process Bonsai and Parabel data")
# print("Transform start")
# updated_readme_description_save_splitted_dataset_as_numpy()
# print("Transform finished")
# updated_readme_description_prepare_omikuji_dataset()
# print("Bonsai and Parabel data preparation finished")


# print("Process LightXML data")
# # print("Transform start")
# # updated_readme_description_save_splitted_dataset_as_numpy()
# # print("Transform finished")
# prepare_lightxml_dataset("dataset/without_lemma_dataset_train.csv", "dataset/without_lemma_dataset_test.csv", "dataset/dataset_labels_ver2.csv")
# print("LightXML data preparation finished")


print("Process ZestXML data")
print("Data splitting start")
zero_shot_data_splitting()
print("Data splitting finished")
save_splitted_zero_shot_dataset_combined_validation_as_numpy()
print("Transform to numpy finished")
prepare_zestxml_dataset()
print("ZestXML data preparation finished")


