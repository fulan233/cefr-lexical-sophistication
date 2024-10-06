import glob
from tqdm import tqdm
from bert_serving.client import BertClient
from tag_sense import *
from utils import *

bc = BertClient()

def tag_text(infile, outfile):
    token_count_total = 0
    # step 1. text preprocessing

    all_sents = text2sents(infile)
    keep_info = []

    for line_num, sent in enumerate(all_sents):
        sent = sent.lower()
        wordlist, poslist = nltk_pos(sent)
        check_wordlist = []
        for word in wordlist:
            if "'" in word:
                check_wordlist.append(abbreconvert(word))
            else:
                check_wordlist.append(word)
        wps = {check_wordlist[i]: poslist[i] for i in range(len(check_wordlist))}
        newsent = ' '.join(check_wordlist)
        if len(wps) >= 1:
            keep_info.append([line_num, newsent, wps])
            # for reliable contextual representations, we only tag sentences with >= 5 tokens
            # for EVP tranning, remove this restriction. hunan edited.

    # step 2. generate contextualized embeddings for the sentences
    input_sents = [k[1] for k in keep_info]
    vec = bc.encode(input_sents, show_tokens=True)
    all_arrays, all_tokens = vec

    # step 3. tag the sentences based on dictionary information
    label_result = {}

    for sid in range(len(all_tokens)):
        # load sentence information
        line_num, sentence, wps = keep_info[sid]
        tokens = all_tokens[sid]
        token_count_total += len(tokens)
        arrays = all_arrays[sid]
        label_result[line_num] = {'tokens': [], 'labels': []}
        token_id = -1

        # tag each token in a sentence
        for i in range(len(tokens)):
            token = tokens[i]
            if token in ['[CLS]', '[SEP]']:
                continue
            elif token.startswith('##'):
                label_result[line_num]['tokens'][-1] += token.replace('##', '')
                continue

            token_id += 1
            label_result[line_num]['tokens'].append(token)
            pos_tag = wps.get(token, 'None')
            token_lm = lemmatize(token, pos_tag)

            # mono word first
            if token in mono_dict.keys():
                CEFR_level = mono_dict.get(token)
                label_result[line_num]['labels'].append([token_id, token + "_" + CEFR_level])

            elif token_lm in mono_dict.keys():
                CEFR_level = mono_dict.get(token_lm)
                label_result[line_num]['labels'].append([token_id, token_lm + "_" + CEFR_level])

            elif token in target_words or token_lm in target_words:
                emb = arrays[i]
                if token_lm in target_words:
                    token = token_lm  # lemma first, token second

                sense_tag = tagSenseFromWordDict(token, pos_tag, emb, False)  # with or without POS information
                # sense_tag = tagSensePosOnly(token, pos_tag) # exp suggested by R2

                if sense_tag:
                    sense_id, simi = sense_tag
                    label_result[line_num]['labels'].append([token_id, sense_id])

            # modified to limit the token to alpha words only
            elif token.isalpha():
                label_result[line_num]['labels'].append([token_id, token + '_' + 'UN'])

    # step 4. save the tagged result
    with open(outfile, 'w', encoding='utf-8') as f:
        for line_num, line in enumerate(all_sents):
            if line_num not in label_result:
                newline = tag_and_lem(line) + '\n'
                f.write(newline)
            else:
                tokenlist = label_result[line_num]['tokens']
                labels = label_result[line_num]['labels']
                for l in labels:
                    token_id, sense_id = l
                    tokenlist[token_id] = sense_id
                newline = ' '.join(tokenlist) + '\n'
                f.write(newline)


if __name__ == '__main__':
    files = glob.glob('./samples/*.txt')
    print(len(files), 'files to be processed')

    for file_num, infile in tqdm(enumerate(files)):
        outfile = infile.replace('samples', 'output')
        tag_text(infile, outfile)
