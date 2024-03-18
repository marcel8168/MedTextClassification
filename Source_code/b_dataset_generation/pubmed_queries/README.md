# EDirect Python Implementation
This directory contains code for querying the PubMed database via NCBI's Entrez Direct (EDirect) using Python. There are also examples of other library functions that can be used for similar queries (but with possible limitations).
</p>

## Table of contents

* [Installation](#installation)
* [Further Information](#further-information)
* [License](#licence)


## Installation and Execution

1. Copy your API key from PubMed (see [How to get API key](https://support.nlm.nih.gov/knowledgebase/article/KA-05317/en-us)) into api_key.txt
2. Optional: Customize the query for your use case in the file query.py. 
The current query returns all articles of journal "Can Vet J" (The Canadian Veterinary Journal) that include an abstract.
3. Start docker engine
4. Build and run the docker container that automatically executes the query.py script:
```shell
cd /Source_code/b_dataset_generation/pubmed_queries
# Docker runs all installations and executes the query.py script
docker compose up --build
```

The requested JSON file will be available in the appropriate directory under ./data.

## Further Information
##### EDirect and E-Utilities
|Topic|Link|
|:-----|:--------|
|PubMed API|https://www.ncbi.nlm.nih.gov/pmc/tools/developers/|
|Entrez Direct|https://www.ncbi.nlm.nih.gov/books/NBK179288/|
|EDirect Installation|https://dataguide.nlm.nih.gov/edirect/install.html|
|ESearch|https://dataguide.nlm.nih.gov/edirect/esearch.html|
|Xtract|https://dataguide.nlm.nih.gov/edirect/xtract.html|
|E-Utilities|https://www.ncbi.nlm.nih.gov/books/NBK25499/|
|Journal IDs|https://ftp.ncbi.nih.gov/pubmed/J_Medline.txt|
##### Library Options
|Topic|Link|
|:-----|:--------|
|URL query|https://github.com/dtoddenroth/medicaleponyms/blob/main/downloadabstracts/pubmedcache.py|
|MetaPub|https://github.com/metapub/metapub|
|PyMed|https://github.com/gijswobben/pymed|
|EntrezPy|https://gitlab.com/ncbipy/entrezpy|

## License

[MIT License](LICENSE) (Marcel Hiltner, 2023)
