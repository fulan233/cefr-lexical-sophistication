import pickle as p
from utils import lemmatize,cosSimilarity,convert_pos

# load dict data
dict_data = './dict/EVP_poly_dict.data'
sense_emb = './dict/EVP_AUG_sense_dict.data'
mono_dict_path = './dict/monodict.data'

with open(dict_data, 'rb') as f:
    target_words = p.load(f)

with open(sense_emb,'rb') as f:
    dic = p.load(f)

with open(mono_dict_path,'rb') as f:
    mono_dict = p.load(f)

print('loaded {} polysemous words and {} sense embeddings.'.format(len(target_words), len(dic)))

def tagSenseFromWordDict(word,pos,word_emb, withPOS=True): 
    '''
    when setting withPOS as True, retrieve the candidate senses of the same pos tag.
    '''
    # retrieve the candidate sense_keys
    sense_dict = target_words.get(word, {})
    if not sense_dict:
        return []
    if withPOS and pos in sense_dict:
        candidates = sense_dict[pos]
    else:
        candidates = []
        for v in sense_dict.values():
            candidates.extend(v)

    # compute the similarities and sort
    simi = {}
    for sid in candidates:
        if sid in dic:
            sense_emb = dic[sid]
            try:
                similarity = cosSimilarity(word_emb, sense_emb)
                simi[sid] = similarity
            except:
                print(sid+" embbeding error")

    if simi:
        sort_simi = sorted(simi.items(), key=lambda d:d[1], reverse=True)
        return sort_simi[0]
    else:
        return []
