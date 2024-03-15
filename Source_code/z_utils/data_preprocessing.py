import itertools
import re
from xml.etree.ElementTree import ElementTree
import pandas as pd
from tqdm import tqdm

from .global_constants import LABELS_MAP


def xml_to_df(xml_file_paths: list[str], preprocess_numbers=False):

    data_sets = [[], []]
    record_sets = []

    tree = ElementTree()

    for i, med_field in enumerate(xml_file_paths):
        med_field_lists = []
        for xml in med_field:
            temp = tree.parse(xml)
            med_field_lists.append(temp.findall('.//Rec'))
        record_sets.append(list(itertools.chain(*med_field_lists)))

    progress_bar = tqdm(range(sum(len(x) for x in record_sets)))

    for i, med_field in enumerate(LABELS_MAP.keys()):
        print(f"Processing medical field: {med_field}")
        labels = LABELS_MAP[med_field]
        for rec in record_sets[i]:
            try:
                common = rec.find('.//Common')
                pmid = common.find('PMID').text
                text_types = [elem.text for elem in common.findall('Type')]
                title = common.find('Title').text
                abstract = common.find('Abstract').text
                mesh_term_list = rec.find('.//MeshTermList')
                mesh_terms = [
                    term.text for term in mesh_term_list.findall('MeshTerm')]
            except Exception as e:
                print(f"An error occurred: {e}")
                print(f"Error occured for PMID: {pmid}")

            data_sets[i].append({'pmid': pmid, "text_types": text_types, 'title': preprocess_text(title, numbers=preprocess_numbers),
                                 'abstract': preprocess_text(abstract, numbers=preprocess_numbers), 'meshtermlist': mesh_terms, 'labels': labels})
            progress_bar.update(1)

    hum_df = pd.DataFrame(data_sets[0])
    vet_df = pd.DataFrame(data_sets[1])

    return hum_df, vet_df


def preprocess_text(text, lower_case=True, special_chars=True, numbers=False):
    text_after_case_processing = text.lower() if lower_case else text

    if special_chars:
        text_after_tab_processing = re.sub(
            r'[\r\n]+', ' ', text_after_case_processing)
        text_after_special_chars_processing = re.sub(
            r'[^\x00-\x7F]+', ' ', text_after_tab_processing)
    else:
        text_after_special_chars_processing = text_after_case_processing

    text_after_number_processing = re.sub(
        r'\d', '', text_after_special_chars_processing) if numbers else text_after_special_chars_processing

    return text_after_number_processing
