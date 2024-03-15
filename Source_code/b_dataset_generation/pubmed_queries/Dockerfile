FROM python:3.8

WORKDIR /data
COPY . /data

RUN apt-get update
RUN pip install -r requirements.txt

# Install EDirect
RUN yes | sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)" \
    && echo "export PATH=/root/edirect:\${PATH}" >> ${HOME}/.bashrc \
    && export PATH=${HOME}/edirect:${PATH} \
    && export NCBI_API_KEY=$(<api_key.txt)

# Run query.py when the container launches
CMD ["/bin/bash", "-c", "source ${HOME}/.bashrc && python query.py"]