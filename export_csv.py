import glob
import re
import pickle
import pandas as pd
import os
import sys
from math import sqrt

with open("./dict/mono_pos.data", "rb") as f:
    mono_pos_dict = pickle.load(f)

# load level data
with open('./dict/target_level.data', 'rb') as f:
    ld = pickle.load(f)

tags = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'UN']

tag_score_dict = {
    'A1': 1,
    'A2': 2,
    'B1': 3,
    'B2': 4,
    'C1': 5,
    'C2': 6,
    'UN': 0
}

plus_feature_type = {"A": ['A1', 'A2'],
                     "B": ['B1', 'B2'],
                     "C": ['C1', 'C2'],
                     "AboveB1": ['B1', 'B2', 'C1', 'C2'],
                     "AboveB2": ['B2', 'C1', 'C2']
                     }

POSTFIX = ["_type_ratio", "_type_root_ratio", "_token_ratio", "_token_root_ratio"]


def extract_tag(sequence, lazy_mode=None):
    pure_tag_sequence = []
    for token in sequence:
        if "#" in token:
            word = token.split('-')[0]
            tag = token.split("#")[1]
            ### for computing modes
            if lazy_mode == 'LazyA1':
                if word in ld and ld[word]['min'] == 'A1':
                    tag = ld[word]['min']
            elif lazy_mode == 'Rand':
                if word in ld:
                    tag = ld[word]['rand']
            elif lazy_mode == 'Min':
                if word in ld:
                    tag = ld[word]['min']

            pure_tag_sequence.append(tag)
        elif "_" in token:
            word, tag = token.split("_")
            pure_tag_sequence.append(tag)

    return pure_tag_sequence

def Is_Content_Token(token):

    if "#" in token:
        # word = token.split('-')[0]
        # tag = token.split("#")[1]
        pos = token.split('_')[0].split('-')[-1]
        if pos in ['noun', 'verb', 'adjective', 'adverb']: # content words onnly
            return True
    elif "_" in token:
        word, tag = token.split("_")
        pos_tags = mono_pos_dict.get(word, 'NA')
        if pos_tags[0] in ['noun', 'verb', 'adjective', 'adverb'] or pos_tags == 'NA': # for EVP unlisted words
            return True

    return False


def mean_sophistication_score(tag_sequence: list):
    total_score = 0
    for tag in tags:
        total_score += tag_sequence.count(tag) * tag_score_dict[tag]
    return total_score


def output_csv(path, outfile, tag_mode):
    files = glob.glob(path)
    indices = {}

    for file in files:
        filename = os.path.split(file)[-1]
        feature_dict = {}
        type_sequence = []
        all_sequence = []

        for line in open(file, "r", encoding='UTF-8'):
            ll = line.strip().split(" ")
            for tag_token in ll:
                if '#' not in tag_token and '_' not in tag_token:
                    continue
                # content words only
                if tag_mode == 'CW' and not Is_Content_Token(tag_token):
                    continue
                all_sequence.append(tag_token)
                if tag_token not in type_sequence:
                    type_sequence.append(tag_token)

        type_length, root_type_length = len(type_sequence), sqrt(len(type_sequence))
        amount, root_amount = len(all_sequence), sqrt(len(all_sequence))

        type_pure_tags_sequence = extract_tag(type_sequence, tag_mode)
        amount_pure_tags_sequence = extract_tag(all_sequence, tag_mode)

        for tag in tags:
            feature_dict[tag + "_type_ratio"] = round((type_pure_tags_sequence.count(tag) / type_length), 3)
            feature_dict[tag + "_type_root_ratio"] = round((type_pure_tags_sequence.count(tag) / root_type_length), 3)
            feature_dict[tag + "_token_ratio"] = round((amount_pure_tags_sequence.count(tag) / amount), 3)
            feature_dict[tag + "_token_root_ratio"] = round((amount_pure_tags_sequence.count(tag) / root_amount), 3)

        for compound_name, tag_sequence in plus_feature_type.items():
            for post in POSTFIX:
                feature_dict[compound_name + post] = round(sum([feature_dict[k + post] for k in tag_sequence]), 3)

        feature_dict['token_mean_score'] = round(mean_sophistication_score(amount_pure_tags_sequence) / amount, 3)
        feature_dict['type_mean_score'] = round(mean_sophistication_score(type_pure_tags_sequence) / type_length, 3)

        # update the EVP indices
        indices[filename] = feature_dict

    # export the results to a csv file
    df_results = pd.DataFrame.from_dict(indices, orient='index')
    df_results.index.name = 'filename'
    df_results.to_csv(outfile)


if __name__ == '__main__':
    tag_mode = 'LazyA1'  # Default setting
    if len(sys.argv) > 1:
        tag_mode = sys.argv[1]
        if tag_mode not in ['AW', 'CW', 'Min', 'LazyA1']:
            print(f'Mode [{tag_mode}] not supported, Please use: AW/CW/Min/LazyA1...')
            exit()
    output_csv('./output/*.txt', f'EVP_indices_{tag_mode}.csv', tag_mode)
