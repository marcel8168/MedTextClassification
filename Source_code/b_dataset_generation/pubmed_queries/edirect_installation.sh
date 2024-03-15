#!/bin/bash

# Download and install EDirect
yes | sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"

# PATH variable assignment in .bashrc file
echo "export PATH=/root/edirect:\${PATH}" >> ${HOME}/.bashrc

# set the PATH for the current terminal session
export PATH=${HOME}/edirect:${PATH}

# API key for higher request rate (10 requests/s)
# for API key create an account via https://account.ncbi.nlm.nih.gov/
export NCBI_API_KEY=$(<api_key.txt)