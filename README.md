# Classification of veterinary subjects in medical literature and clinical summaries

Source code of the Master's Thesis in Computer Science at the Friedrich-Alexander-University (FAU). 
Author: Marcel Hiltner

## Table of contents

* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [License](#licence)

## Installation and Execution
Note: All results of the master thesis were obtained in Kaggle Notebooks. Local execution of the code may lead to deviating results.
To install, follow the steps below:

1. Clone the repository:
```shell
git clone https://github.com/marcel8168/medtextclassification medtextclassification
```
2. Install the requirements
```shell
cd medtextclassification
pip install .
```
3. To be able to load datasets and models used in this repository first set username and api-key from kaggle (can be created in kaggle profile settings)
```shell
# linux
export KAGGLE_USERNAME=xxxxxxxxxxxxxx
export KAGGLE_KEY=xxxxxxxxxxxxxx

# windows
SET KAGGLE_USERNAME=xxxxxxxxxxxxxx
SET KAGGLE_KEY=xxxxxxxxxxxxxx
```
4. Optional: For querying PubMed first copy your API key from PubMed (see [How to get API key](https://support.nlm.nih.gov/knowledgebase/article/KA-05317/en-us)) into api_key.txt



## Repository Structure
<pre>
├── Source_code                             | Folder for the source code
    ├── a_problem_analysis                  | Folder for problem analysis notebooks
        ├── analysis_pmc_patients.ipynb     | Notebook analyzing PMC patients data
        └── problem_analysis.ipynb          | Notebook for general problem analysis
    ├── b_dataset_generation                | Folder for dataset generation
        ├── data                            | Folder containing data queried from PubMed
        │   ├── human_medical_data          | Folder for human medical data
        │   │   ├── BMJ_data.xml            | XML file for for texts of journal BMJ
        │   │   └── NEJM_data.xml           | XML file for for texts of journal NEJM
        │   └── veterinary_medical_data     | Folder for veterinary medical data
        │       ├── Animals_data.xml        | XML file for texts of journal Animals
        │       └── ...                     | Other XML files for veterinary journal texts
        ├── pubmed_queries                  | Folder for PubMed queries
        │   ├── api_key.txt                 | API key for PubMed
        │   ├── docker-compose.yaml         | Docker Compose configuration
        │   ├── Dockerfile                  | Dockerfile for PubMed setup
        │   ├── edirect.py                  | Python script for EDirect setup
        │   ├── edirect_installation.sh     | Shell script for EDirect installation
        │   ├── library_options.ipynb       | Notebook for library options
        │   ├── query.py                    | Python script for PubMed queries
        │   └── requirements.txt            | Requirements file for PubMed setup
        └── dataset_generation.ipynb        | Notebook for dataset generation
    ├── c_model_training_fine_tuning        | Folder for model training and fine-tuning
        ├── plm_fine_tuning.ipynb           | Notebook for PLM fine-tuning
        └── svm_training.ipynb              | Notebook for SVM training
    ├── d_model_testing                     | Folder for model testing
        ├── plm_testing.ipynb               | Notebook for PLM testing
        └── svm_testing.ipynb               | Notebook for SVM testing
    ├── e_model_interpretation              | Folder for model interpretation
        ├── rare_animals.ipynb              | Notebook for analysis of texts containing rare animals
        ├── svm_coefficients.ipynb          | Notebook for SVM coefficients analysis
        └── word_importance.ipynb           | Notebook for word importance analysis
    ├── f_others                            | Folder for other analyses
        └── hardware_analysis.ipynb         | Notebook for hardware analysis
    └── z_utils                             | Folder for utility scripts and classes
        ├── BERTClassifier.py               | Python script for BERT classifier
        ├── BlueBERTClassifier.py           | Python script for BlueBERT classifier
        ├── data_preparing.py               | Python script for data preparation
        ├── data_preprocessing.py           | Python script for data preprocessing
        ├── Dataset.py                      | Python script for dataset class
        ├── DeBERTaClassifier.py            | Python script for DeBERTa classifier
        ├── evaluate.py                     | Python script for model evaluation
        ├── global_constants.py             | Python script for global constants
        ├── lemmatize.py                    | Python script for text lemmatization
        ├── loss_fn.py                      | Python script for loss function
        ├── plot.py                         | Python script for plotting
        ├── predict.py                      | Python script for prediction
        ├── RoBERTaClassifier.py            | Python script for RoBERTa classifier
        ├── train.py                        | Python script for PLM training
        └── XLNetClassifier.py              | Python script for XLNet classifier
├── README.md                               | Readme file
├── requirements.txt                        | Requirements file
└── setup.py                                | Setup file
</pre>

## License
[MIT License](LICENSE) (Marcel Hiltner, 2023)