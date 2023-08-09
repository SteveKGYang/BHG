# Bipartite Heterogeneous Graph for Emotional Reasoning

This repository contains our source code and data for the CIKM 2023 accepted
paper: [A Bipartite Graph is All We Need for Enhancing Emotional Reasoning with
Commonsense Knowledge]().

Here is an overview of the model:

![Image text](https://github.com/SteveKGYang/BHG/blob/main/BHG_model.png)


## Preparation

* Set up the Python 3.7 environment, and build the dependencies with the following code:
pip install -r requirements.txt

* Install the torch-geometric package from [this link](https://pytorch-geometric.readthedocs.io/en/latest/).

* **You can download our extracted training data and knowledge features**
for each knowledge source with the following links: [COMET<sub>2019</sub>](https://drive.google.com/file/d/12EiLPGu6gheQs2wZq-ifux1DAgensgfg/view?usp=sharing),
[COMET<sub>2020</sub>](https://drive.google.com/file/d/1cAQ3zk-fVrlWI7o1ACX5B758bsyzIxWA/view?usp=sharing),
[Conceptnet](https://drive.google.com/file/d/1vrd1TI4utbgAWoXTtR_hsj3rO3rGzdKd/view?usp=sharing).
Due to storage limit, we don't provide extracted data for DailyDialog. You can use the
following data to re-extract knowledge data for all datasets.

* **Or you can build data from scratch**. Firstly, set up the extraction environments for each knowledge source.

   1. For COMET<sub>2019</sub>, The related code is in ./comet/. First download the pre-trained COMET models from
      [this link](https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB)
      and put it under ./comet/pretrained_models/ directory.
   
   2. For COMET<sub>2020</sub>: Download the pre-trained model with the script: download_model.sh
   
   3. For Conceptnet, download our filtered english version of conceptnet 
      from [this link](https://drive.google.com/file/d/1i87UQLm3FZilp9J2CfqJLPGl3sJQ79v4/view?usp=sharing) and put it under
      ./conceptnet/.

   Secondly, download the original ERC datasets from 
   [this link](https://drive.google.com/file/d/1b_ihQYKTAsO67I5LULMbMFBrDgat8bQN/view?usp=sharing)
   We also directly provide the original RECCON dataset in this repository (/RECCON_data).
   Put the downloaded data under the directory of each data source.

   Thirdly, run the script to extract the knowledge features:
   1. For COMET<sub>2019</sub>, modify and run comet_extract_origin.py. This script is adapted from
   the [COMET source code](https://github.com/atcbosselut/comet-commonsense/tree/master)
   2. For COMET_2020, modify and run generate_knowledge.py. This script is adapted from the
   source code of [Zhao et al.](https://github.com/circle-hit/KBCIN).
   3. For Conceptnet, modify and run preprocess_conceptnet.py, then run main.py.

## Training:

For training on ERC datasets, we use IEMOCAP as an exmaple:

Training with COMET<sub>2020</sub>:
```
python main.py --DATASET IEMOCAP --model_checkpoint roberta-large --NUM_TRAIN_EPOCHS 10 --BATCH_SIZE 16 --model_save_dir ./model_save_dir/IEMOCAP --mode train --SEED 42 --ROOT_DIR ./bart_comet_enhanced_data/ --CONV_NAME hgt --COMET_HIDDEN_SIZE 1024 --CUDA
```
Training with COMET<sub>2019</sub>:
```
python main.py --DATASET IEMOCAP --model_checkpoint roberta-large --NUM_TRAIN_EPOCHS 10 --BATCH_SIZE 16 --model_save_dir ./model_save_dir/IEMOCAP --mode train --SEED 42 --ROOT_DIR ./comet_origin_enhanced_data/ --CONV_NAME multidim_hgt --COMET_HIDDEN_SIZE 768 --CUDA
```
Training with Conceptnet:
```
python main.py --DATASET IEMOCAP --model_checkpoint roberta-large --NUM_TRAIN_EPOCHS 10 --BATCH_SIZE 16 --model_save_dir ./model_save_dir/IEMOCAP --mode train --SEED 42 --ROOT_DIR ./conceptnet_enhanced_data/ --CONV_NAME multidim_hgt --COMET_HIDDEN_SIZE 768 --CUDA
```

Training on other ERC datasets are similar.


For training on CEE dataset RECCON:

Training with COMET<sub>2020</sub>:
```
python main.py --DATASET RECCON --model_checkpoint roberta-large --alpha 0.8 --NUM_TRAIN_EPOCHS 10 --BATCH_SIZE 1 --model_save_dir ./model_save_dir/RECCON --mode train --LR 3e-6 --SEED 42 --ROOT_DIR ./bart_comet_enhanced_data/ --CONV_NAME hgt --COMET_HIDDEN_SIZE 1024 --CUDA
```
Training with COMET<sub>2019</sub>:
```
python main.py --DATASET RECCON --model_checkpoint roberta-large --alpha 0.8 --NUM_TRAIN_EPOCHS 10 --BATCH_SIZE 1 --model_save_dir ./model_save_dir/RECCON --mode train --LR 3e-6 --SEED 42 --ROOT_DIR ./comet_origin_enhanced_data/ --CONV_NAME multidim_hgt --COMET_HIDDEN_SIZE 768 --CUDA
```
Training with Conceptnet:
```
python main.py --DATASET RECCON --model_checkpoint roberta-large --alpha 0.8 --NUM_TRAIN_EPOCHS 10 --BATCH_SIZE 1 --model_save_dir ./model_save_dir/RECCON --mode train --LR 3e-6 --SEED 42 --ROOT_DIR ./conceptnet_enhanced_data/ --CONV_NAME multidim_hgt --COMET_HIDDEN_SIZE 768 --CUDA# BHG
```

## Citation

Please cite the paper as follows:
