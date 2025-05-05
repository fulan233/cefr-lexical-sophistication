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

def extract_type_sequence(sequence,lazy_mode = None):
    ''''for MA mode type calculation'''
    typelist = []
    pure_tag_sequence = []
    for token in sequence:
        if "#" in token:
            word = token.split('-')[0]
            type = token.split("#")[0]
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

            if type not in typelist:
                pure_tag_sequence.append(tag)
                typelist.append(type)
            else:
                pure_tag_sequence.append('PLACEHOLDER')
        elif "_" in token:
            word, tag = token.split("_")
            if word not in typelist:
                pure_tag_sequence.append(tag)
                typelist.append(word)
            else:
                pure_tag_sequence.append('PLACEHOLDER')

    return pure_tag_sequence



def Is_Content_Token(token):

    if "#" in token:
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


def output_csv(path, outfile, tag_mode,window_size = 100):
    files = glob.glob(path)
    indices = {}

    for file in files:
        filename = os.path.split(file)[-1]
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
        all_word_type_pure_tags_sequence = extract_type_sequence(all_sequence,tag_mode)

        if window_size <= 0:
            raise ValueError("Window size error.")
        elif window_size > len(all_word_type_pure_tags_sequence):
            print("The sliding window size has been set to greater than the sequence length and has been reset to half of the sequence length.")
            window_size = int(len(all_word_type_pure_tags_sequence)/2)

        temp_feature_dict = {}

        for i in range(len(all_word_type_pure_tags_sequence) - window_size + 1):

            window = all_word_type_pure_tags_sequence[i:i + window_size]
            window_type_amount = len([item for item in all_word_type_pure_tags_sequence if item != 'PLACEHOLDER'])

            for tag in tags:
                if "MA_" + tag + "_type" not in temp_feature_dict.keys():
                    temp_feature_dict["MA_" + tag + "_type"] = [round((window.count(tag) / window_type_amount),3)]
                else:
                    temp_feature_dict["MA_" + tag + "_type"].append(round((window.count(tag) / window_type_amount),3))

            for compound_name, tag_sequence in plus_feature_type.items():
                if "MA_" + compound_name + "_type" not in temp_feature_dict.keys():
                    temp_feature_dict["MA_" + compound_name + "_type"] = [sum([temp_feature_dict["MA_" + k + "_type"][i] for k in tag_sequence])]
                else:
                    temp_feature_dict["MA_" + compound_name + "_type"].append(sum([temp_feature_dict["MA_" + k + "_type"][i] for k in tag_sequence]))

        for i in range(len(amount_pure_tags_sequence) - window_size + 1):
            window = amount_pure_tags_sequence[i:i + window_size]
            for tag in tags:
                if "MA_" + tag + "_token" not in temp_feature_dict.keys():
                    temp_feature_dict["MA_" + tag + "_token"] = [window.count(tag)/window_size]
                else:
                    temp_feature_dict["MA_" + tag + "_token"].append(window.count(tag)/window_size)

            for compound_name, tag_sequence in plus_feature_type.items():
                if "MA_" + compound_name + "_token" not in temp_feature_dict.keys():
                    temp_feature_dict["MA_" + compound_name + "_token"] = [sum([temp_feature_dict["MA_" + k + "_token"][i] for k in tag_sequence])]
                else:
                    temp_feature_dict["MA_" + compound_name + "_token"].append(sum([temp_feature_dict["MA_" + k + "_token"][i] for k in tag_sequence]))

        indices[filename] = {k: round(sum(v) / len(v), 3) for k, v in temp_feature_dict.items()}
        indices[filename]['token_mean_score'] = round(mean_sophistication_score(amount_pure_tags_sequence) / amount,3)
        indices[filename]['type_mean_score'] = round(mean_sophistication_score(type_pure_tags_sequence) / type_length, 3)

    # export the results to a csv file
    df_results = pd.DataFrame.from_dict(indices, orient='index')
    df_results.index.name = 'filename'
    df_results.to_csv(outfile)




if __name__ == '__main__':
    tag_mode = 'LazyA1'  # Default setting
    window_size = 100 # Default setting
    if len(sys.argv) > 1:
        tag_mode = sys.argv[1]
        if tag_mode not in ['AW', 'CW', 'Min', 'LazyA1','Rand']:
            print(f'Mode [{tag_mode}] not supported, Please use: AW/CW/Min/LazyA1...')
            exit()

        if len(sys.argv) > 2:
            window_size = int(sys.argv[2])



    print(f'Tag mode: {tag_mode}, Window size: {window_size}')
    output_csv('./output/*.txt', f'EVP_indices.csv',tag_mode = tag_mode, window_size = window_size)

