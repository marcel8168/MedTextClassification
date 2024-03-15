#!/usr/bin/env python3
# https://www.ncbi.nlm.nih.gov/books/NBK179288/#chapter6.Python_Integration

import sys
import os
import shutil

import pandas as pd
import edirect
from xml.etree.ElementTree import fromstring, ElementTree


# query constants
# journal abbreviation and ISSN available under https://ftp.ncbi.nih.gov/pubmed/J_Medline.txt
# note that special characters in the query term lead to syntax errors

# JOURNAL = "N Engl J Med"
# JOURNAL_ISSN = "1533-4406"
# JOURNAL = "Animals"
# JOURNAL_ISSN = "2076-2615"
# JOURNAL = "J Vet Med Sci"
# JOURNAL_ISSN = "1347-7439"
# JOURNAL = "Case Rep Vet Med"
# JOURNAL_ISSN = "2090-701X"
# JOURNAL = "Open Vet J"
# JOURNAL_ISSN = "2218-6050"
# JOURNAL = "Front Vet Sci"
# JOURNAL_ISSN = "2297-1769"
# JOURNAL = "Vet Sci"
# JOURNAL_ISSN = "2306-7381"
# JOURNAL = "Vet Med Sci"
# JOURNAL_ISSN = "2053-1095"
# JOURNAL = "J Small Anim Pract"
# JOURNAL_ISSN = "1748-5827"
# JOURNAL = "J Am Anim Hosp Assoc"
# JOURNAL_ISSN = "1547-3317"
# JOURNAL = "BMJ Case Rep"
# JOURNAL_ISSN = "1757-790X"
# JOURNAL = "J Vet Intern Med"
# JOURNAL_ISSN = "1939-1676"
# JOURNAL = "J Vet Diagn Invest"
# JOURNAL_ISSN = "1943-4936"
# JOURNAL = "J Am Vet Med Assoc"
# JOURNAL_ISSN = "1943-569X"
# JOURNAL = "Vet Pathol"
# JOURNAL_ISSN = "1544-2217"
# JOURNAL = "Can Vet J"
# JOURNAL_ISSN = "0008-5286"
# JOURNAL = "J Zoo Wildl Med"
# JOURNAL_ISSN = "1042-7260"
# JOURNAL = "Am J Trop Med Hyg"
# JOURNAL_ISSN = "1042-7260"
JOURNAL = "Can Vet J"
JOURNAL_ISSN = "0008-5286"

TARGET_DIRECTORY = "../data/"
OUTPUT_FILE = "CanVetJ_data.xml"


xtract_path = shutil.which('xtract')

if xtract_path is not None:
    sys.path.insert(1, os.path.dirname(xtract_path))
else:
    print("xtract not found in the PATH.")

if not os.path.exists(TARGET_DIRECTORY):
    os.makedirs(TARGET_DIRECTORY)

# for more information regarding the query see the guide: https://dataguide.nlm.nih.gov/edirect/xtract.html
# or watch the webinar videos on YouTube: https://www.youtube.com/playlist?list=PL7dF9e2qSW0a6zx-yGMJvY6mcwQz_Vx4b
query = f'esearch -db pubmed -query "{JOURNAL_ISSN}[JOUR] AND {JOURNAL}[JOUR]" | \
    efetch -format xml | \
    xtract -set Set \
        -rec Rec -pattern PubmedArticle \
        -if AbstractText \
        -tab "\n" -sep "," \
        -block PubmedArticle -pkg Common \
            -wrp PMID -element MedlineCitation/PMID \
            -wrp Type -element PublicationType \
            -wrp Title -element Article/ArticleTitle \
            -wrp Abstract -element AbstractText \
        -block MeshHeadingList -pkg MeshTermList \
            -wrp MeshTerm -element MeshHeading/DescriptorName'

print("Query is being processed..")
res = edirect.pipeline(query)

if not res:
    print("No results available.")
else:
    res_xml = fromstring(res)
    ElementTree(res_xml).write(TARGET_DIRECTORY + OUTPUT_FILE)
    print(f"Query output written into {TARGET_DIRECTORY + OUTPUT_FILE}")
