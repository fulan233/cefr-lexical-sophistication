import glob
import re
import pickle
import pandas as pd
import os
from math import sqrt

with open("puncs.data", "rb") as f1:
    punc = pickle.load(f1)
nums = re.compile(r'[0-9]', re.S)

tags = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'UN']

tag_score_dict={
    'A1':1,
    'A2':2,
    'B1':3,
    'B2':4,
    'C1':5,
    'C2':6,
    'UN':0
}

plus_feature_type = {"A": ['A1', 'A2'],
                     "B": ['B1', 'B2'],
                     "C": ['C1', 'C2'],
                     "AboveB1": ['B1', 'B2', 'C1', 'C2'],
                     "AboveB2": ['B2', 'C1', 'C2']
                    }

POSTFIX = ["_type_ratio", "_type_root_ratio", "_token_ratio", "_token_root_ratio"]


def extract_tag(sequence): 
    pure_tag_sequence = []
    for token in sequence:
        if "#" in token:
            pure_tag_sequence.append(token.split("#")[1])

        elif "_" in token:
            tag = token.split("_")[1]
            ttoken = token.split("_")[0]
            if tag == "UN":
                if ttoken in punc:
                    continue
                elif len(re.findall(nums, ttoken)) > 0:
                    continue
                else:
                    pure_tag_sequence.append('UN')
            else:
                pure_tag_sequence.append(tag)
        else:
            print(token)
    return pure_tag_sequence



def mean_sophisitication_score(tag_sequence:list):
    total_score = 0
    for tag in tags:
        total_score += tag_sequence.count(tag) * tag_score_dict[tag]
    return total_score



def output_csv(path, outputname):
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
                all_sequence.append(tag_token)
                if tag_token not in type_sequence:
                    type_sequence.append(tag_token)


        type_length, root_type_length = len(type_sequence), sqrt(len(type_sequence))
        amount, root_amount = len(all_sequence), sqrt(len(all_sequence))

        type_pure_tags_sequence = extract_tag(type_sequence)  
        amount_pure_tags_sequence = extract_tag(all_sequence)

        # feature_dict['ntokens'] = amount
        # feature_dict['ntypes'] = type_length

        for tag in tags:
            feature_dict[tag + "_type_ratio"] = round((type_pure_tags_sequence.count(tag) / type_length), 3)
            feature_dict[tag + "_type_root_ratio"] = round((type_pure_tags_sequence.count(tag) / root_type_length), 3)
            feature_dict[tag + "_token_ratio"] = round((amount_pure_tags_sequence.count(tag) / amount), 3)
            feature_dict[tag + "_token_root_ratio"] = round((amount_pure_tags_sequence.count(tag) / root_amount), 3)

        for compound_name, tag_sequence in plus_feature_type.items():
            for post in POSTFIX:
                feature_dict[compound_name + post] = round(sum([feature_dict[k + post] for k in tag_sequence]),3)  
        
        feature_dict['token_mean_score'] = round(mean_sophisitication_score(amount_pure_tags_sequence)/amount,3)
        feature_dict['type_mean_score'] = round(mean_sophisitication_score(type_pure_tags_sequence)/type_length,3)

        # update the EVP indices
        indices[filename] = feature_dict

    # export the results to a csv file
    df_results = pd.DataFrame.from_dict(indices, orient='index')
    df_results.index.name = 'filename'
    df_results.to_csv(outputname)


if __name__ == '__main__':

    output_csv('./output/*.txt', 'evp-indices.csv')

