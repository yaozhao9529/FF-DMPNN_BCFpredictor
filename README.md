# FF-DMPNN_BCFpredictor: Predicting Bioconcentration Factors of Per- and Polyfluoroalkyl Substances

This repository contains the code and data for the paper "Predicting bioconcentration factors of per- and polyfluoroalkyl substances using a directed message passing neural network with multimodal feature fusion".  

## Overview  

Per- and polyfluoroalkyl substances (PFASs) are persistent environmental contaminants of increasing concern due to their widespread use, environmental persistence, and potential toxicity. Bioconcentration factors (BCFs) are critical parameters for assessing the environmental risks of these chemicals. This project introduces a novel feature-fused directed message passing neural network (FF-DMPNN) model that integrates molecular graph representations, physicochemical descriptors, and bioassay data to predict BCF values with high accuracy.

## Installation  

## Clone the repository  
git clone https://github.com/yaozhao9529/FF-DMPNN_BCFpredictor.git  
cd FF-DMPNN_BCFpredictor

## Create and activate a conda environment (optional but recommended)  
conda create -n pfas-bcf python=3.8
conda activate pfas-bcf  

## Install dependencies  
pip install -r requirements.txt

## Data Description
BCF_model_dataset.csv: Contains the bioconcentration factor values for 1672 diverse chemicals, with SMILES structures and experimentally measured BCF values.
BCF_with_descriptors.csv: The BCF dataset augmented with 99 key features, including 91 calculated molecular descriptors from RDKit and 8 bioassays.
descriptor_names.txt: Names of the 91 key physicochemical descriptors screened.
Assay_dataset.csv: Bioassay dataset for model training, including reporter gene assay results for nuclear receptors and enzyme activity assays.
External_dataset.csv: Independent dataset of 18 PFASs for external validation.
External_dataset_with_descriptors.csv: Corresponding 99 key features of 18 PFASs.
PFAS_database.csv: A comprehensive PFAS database, consisting of 13591 chemicals published in the EPA PFAS Structure Lists, with model predicted logBCF.

## Usage
### 1. Feature Extraction
Extract molecular descriptors from chemical structures:

jupyter notebook code/chemicals_descriptor extract.ipynb

### 2. Bioassay Prediction
Predict bioactivity for compounds using pre-trained models:

python code/Assay_prediction.py  
This script uses graph neural network models to predict bioactivity for CAR, PXR, CYP3A4, PPARδ, CYP2C9, and PPARγ assays.
jupyter notebook code/OATP_FABP_prediction.ipynb
This script uses random forest models to predict bioactivity for OATP and FABP4 assays.

### 3. Model Training
Train the FF-DMPNN model with combined features:

python code/FF-DMPNN_BCF_train.py
The training script includes:

Data preprocessing
Feature integration (molecular graphs, descriptors, and bioassay data)
Model training with 5-fold cross-validation
Performance evaluation

### 4. BCF Prediction
Predict BCF values for new compounds:

python code/FF-DMPNN_BCF_prediction.py

## Model Architecture
The FF-DMPNN model integrates three types of features:

Molecular graphs: Captures detailed atomic connectivity and bond relationships
Physicochemical descriptors: Represents molecular properties like molecular weight, SLogP, etc.
Bioassay information: Incorporates biological activity data from key assays (CAR, PXR, CYP3A4, PPARδ, CYP2C9, PPARγ, OATP, FABP4)

## Acknowledgments
The Chemprop package for the base DMPNN implementation
The open-source chemical informatics community for providing essential tools

## Contact
For questions or collaborations, please contact: Jingzhi Yao (yaozhao9529@163.com)
