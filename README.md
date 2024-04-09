# Classification of veterinary subjects in medical literature and clinical summaries

Source code of the Master's Thesis in Computer Science at the Friedrich-Alexander-University (FAU).

Author: Marcel Hiltner

Note: Due to the storage space, all trained or fine-tuned models and the data set are stored on kaggle. However, these can be loaded via the kaggle API, as implemented in the code.

## Table of contents

* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Links to models and dataset](#further-links)
* [License](#license)

## Repository Structure
<pre>
├── Source_code                             | <b>Folder for the source code</b>
    ├── <b>a_problem_analysis</b>                  | <b>Folder for problem analysis</b>
        ├── analysis_pmc_patients.ipynb     | Notebook analyzing PMC patients data
        └── problem_analysis.ipynb          | Notebook for general problem analysis
    ├── <b>b_dataset_generation</b>                | <b>Folder for dataset generation</b>
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
    ├── <b>c_model_training_fine_tuning</b>        | <b>Folder for model training and fine-tuning</b>
        ├── plm_fine_tuning.ipynb           | Notebook for PLM fine-tuning
        └── svm_training.ipynb              | Notebook for SVM training
    ├── <b>d_model_testing</b>                     | <b>Folder for model testing</b>
        ├── plm_testing.ipynb               | Notebook for PLM testing
        └── svm_testing.ipynb               | Notebook for SVM testing
    ├── <b>e_model_interpretation</b>              | <b>Folder for model interpretation</b>
        ├── rare_animals.ipynb              | Notebook for analysis of texts containing rare animals
        ├── svm_coefficients.ipynb          | Notebook for SVM coefficients analysis
        └── word_importance.ipynb           | Notebook for word importance analysis
    ├── <b>f_others</b>                            | <b>Folder for other analyses</b>
        └── hardware_analysis.ipynb         | Notebook for hardware analysis
    └── <b>z_utils</b>                             | <b>Folder for utility scripts and classes</b>
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

## Installation
Note: All results of the master thesis were obtained in Kaggle Notebooks. Local execution of the code may lead to deviating results.
To install, follow the steps below:

1. Clone the repository:
```shell
git clone https://github.com/marcel8168/medtextclassification medtextclassification
```
2. Create a virtual environment
```shell
cd medtextclassification
python -m venv venv
venv\Scripts\activate.bat
```
3. Install PyTorch for computations on CUDA (see [How to install PyTorch](https://pytorch.org/)). Select CUDA as compute platform.
4. Install the requirements
```shell
pip install -e .
pip install -r requirements.txt
```
5. To be able to load datasets and models used in this repository first set username and API key from kaggle (see [How to get API key](https://github.com/Kaggle/kaggle-api?tab=readme-ov-file#api-credentials))
```shell
# linux
export KAGGLE_USERNAME=xxxxxxxxxxxxxx
export KAGGLE_KEY=xxxxxxxxxxxxxx

# windows
SET KAGGLE_USERNAME=xxxxxxxxxxxxxx
SET KAGGLE_KEY=xxxxxxxxxxxxxx
```
6. Optional: For querying PubMed first copy your API key from PubMed (see [How to get API key](https://support.nlm.nih.gov/knowledgebase/article/KA-05317/en-us)) into api_key.txt

## Further links
| Description |  Link  |
|:-----|:--------:|
| BERT Model   | [Link](https://www.kaggle.com/models/marcelhiltner/bert-base-uncased-pubmed) |
| RoBERTa Model   | [Link](https://www.kaggle.com/models/marcelhiltner/roberta-base-pubmed)  |   
| DeBERTa Model   | [Link](https://www.kaggle.com/models/marcelhiltner/deberta-base-pubmed) | 
| BlueBERT Model | [Link](https://www.kaggle.com/models/marcelhiltner/bluebert-large-pubmed) |
| XLNet Model | [Link](https://www.kaggle.com/models/marcelhiltner/xlnet-large-pubmed) |
| SVM Model | [Link](https://www.kaggle.com/models/marcelhiltner/svm-linear-pubmed) |
| Dataset | [Link](https://www.kaggle.com/datasets/marcelhiltner/pubmed-human-veterinary-medicine-classification) |

## License
[MIT License](LICENSE) (Marcel Hiltner, 2023)
