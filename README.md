# XML Models Utilization

## Replication package for Extreme Multi-label Classification for Automated Topic Recommendation from GitHub Repository

## How to Use
This repository's folders (LightXML, Omikuji, and zestxml) each include the implementation of several XML techniques that can be used with the subject recommendation data. Functions to prepare data for LightXML, Omikuji (Bonsai and Parabel), and ZestXML can be found in the DataPreparation folder. As we already include the preprocessed dataset inside the XML models folder, you do not need to run the script in DataPreparation.

## LightXML
### Environment setup
- For easier environment setting, please use conda env
- Install NVidia Apex (https://github.com/NVIDIA/apex)
- Install the requirements listed in requirements.txt
For the detailed information, please refer to https://github.com/kongds/LightXML/

### Training and Evaluation
- Please unzip the 'combined-topic-data.zip' from the LightXML/dataset folder
- Then run "run.sh" script for training and evaluation
  > ./run.sh combined-topic-data
- You can also run it directly from LightXML/src/main.py for training and LightXML/src/ensemble.py for prediction 

## Omikuji (i.e., Bonsai and Parabel)

### Environment Setup
- Install the Python binding of Omikuji for the model prediction:
   > pip install omikuji
- For the model training, we use the Rust implementation tha available in Cargo:
   > cargo install omikuji --features cli --locked

For detailed information, please refer to https://github.com/tomtung/omikuji/

### Training and Evaluation Process
- After Omikuji is successfully installed from Cargo, we can use the following command to train a model:

- **Parabel Model**
   > `omikuji train --model_path model_output_path --min_branch_size 2  --n_trees 3 dataset/train.txt`
- **Bonsai Model**
  > `omikuji train --cluster.unbalanced --model_path model_output_path --n_trees 3 dataset/train.txt`
- For the evaluation, please run omikuji_predict.py with the model_path and test_data_path:
  > python omikuji_predict model_output_path dataset/test.txt

## ZestXML
### Environment setup
- For easier environment setting, please use conda env
- Build modules in zestxml folder using "make" command:
  > make
- Install the requirements listed in requirements.txt
- Install pyxclib for evaluation
  > git clone https://github.com/kunaldahiya/pyxclib.git
  > 
  > cd pyxclib
  > 
  > python3 setup.py install --user
  > 
For the detailed information, please refer to https://github.com/nilesh2797/zestxml

### Training and Evaluation
- Please unzip the 'czero_shot_dataset_combined.zip' from the zestxml/dataset folder
- Then run "run.sh" script with dataset path and model name for training and prediction
  > ./run.sh dataset/zero_shot_dataset_combined/zestxml train topic_recommendation
  > 
  > ./run.sh dataset/zero_shot_dataset_combined/zestxml predict topic_recommendation
- The model generated is inside the dataset_path/Results/model_name
- For getting the metrics result, please run topic_F1_metrics:
  > python topic_F1_metrics.py topic_recommendation dataset/zero_shot_dataset_combined/zestxml

### Explanation Visualization
- After training and evaluation, you can get the explanation why ZestXML produce those recommendation by running the jupyter notebook from zestxml/explanation_zestXML.ipynb

## Dataset Preparation Folder
**data_preparation.py**: contains functions to prepare data for the XML models setting. After running this code, you need to move folder from dataset into the XML dataset folder.

**dataset folder**: contains the dataset of the topic data, you need to unzip the "dataset.zip". The dataset already splitted into train and testing following the experiment for the study.
