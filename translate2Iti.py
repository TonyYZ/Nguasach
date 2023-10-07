import transformers
from transformers import pipeline, BertForMaskedLM, BertModel, AutoTokenizer, AutoModel
import numpy as np
from transformers import BertTokenizer
import torch
# import transPhone
import torch.nn as nn
from lemminflect import getInflection

# Load a pre-trained language model
fill_mask = pipeline("fill-mask", model="bert-base-cased", top_k=10)

# Words without great value
skip_words = ['it', 'he', 'she', 'one', 'I', 'object', 'other', '.', 'is']

verbs = ['clean', 'cook', 'be', 'become', 'do', 'go', 'come', 'enter', 'exit', 'wait', 'stay', 'leave', 'return', 'think', 'consider', 'decide', 'choose', 'feel', 'see', 'hear', 'touch', 'smell', 'taste', 'say', 'know', 'learn', 'teach', 'repeat']
nouns = ['cooking', 'pot', 'food', 'stove', 'chimney', 'floor']

def fine_tune():
    # Load the pre-trained BERT model
    model_name = 'bert-base-uncased'
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    model = transformers.BertModel.from_pretrained(model_name)

    # Define the new trigram tokens
    trigrams = ['坤卦', '艮卦', '坎卦', '巽卦', '中卦', '震卦', '離卦', '兌卦', '乾卦']
    trigram_features = [['void', 'empty', 'static', 'motionless', 'sparse', 'few', 'clear', 'clean', 'quiet', 'silent'],
                 ['return', 'retreat', 'come', 'backward', 'descend', 'cover', 'limitation', 'constrain'],
                 ['occupy', 'middle', 'center', 'source', 'single', 'radiate', 'separate', 'disperse'],
                 ['leak', 'seep', 'permeates', 'shed', 'erode', 'dilute', 'discard', 'depletion'],
                 ['adapt', 'mixed', 'flexible', 'vague'],
                 ['depart', 'advance', 'go', 'forward', 'rise', 'ascend', 'support', 'base', 'carry', 'load'],
                 ['merge', 'combine', 'enclose', 'surround', 'wrap', 'juxtaposition', 'parallel', 'pair'],
                 ['express', 'show', 'release', 'reveal', 'emit', 'give', 'issuing', 'deliver', 'swell'],
                 ['numerous', 'tumultuous', 'energetic', 'full']
                 ]
    # Add the trigram tokens to the vocabulary
    tokenizer.add_tokens(trigrams)
    model.resize_token_embeddings(len(tokenizer))

    # Fine-tune the model on a text corpus
    corpus = [trigram + ' means ' + ', '.join(trigram_features[i]) + '.' for i, trigram in enumerate(trigrams)]

    print(corpus)
    encoded_corpus = [tokenizer.encode(text, add_special_tokens=True) for text in corpus]

    # set max sequence length to the length of the longest sequence
    max_length = max(len(seq) for seq in encoded_corpus)
    # pad the sequences with the 0 index to the max length
    encoded_corpus = [seq + [0] * (max_length - len(seq)) for seq in encoded_corpus]

    # print(encoded_corpus)
    inputs = torch.tensor(encoded_corpus)
    outputs = model(inputs)
    print(outputs)

fine_tune()

def fill_sentence(sentence):
    score_sum = 0
    fillings = []
    mask_num = sentence.count('[MASK]')
    # Generate multiple words to fill in the blanks
    for i in range(mask_num):
        results = fill_mask(sentence)
        # print(sentence, results, i, mask_num - 1)

        if i < mask_num - 1:
            result = results[0]
        else:
            result = results
        j = 0
        # print(result)
        for j in range(len(result)):
            if result[j]['token_str'] not in skip_words and result[j]['token_str'] not in fillings:
                break  # this word can be chosen
        choice = result[j]
        score_sum += choice['score']
        fillings.append(choice['token_str'])
        sentence = choice['sequence'].replace('[CLS] ', '').replace(' [SEP]', '')

    # print(fillings)

    for filling in fillings:
        sentence = sentence.replace('[MASK]', filling, 1)

    print(sentence, score_sum)
    return sentence, score_sum, fillings

# fill_sentence("The spatial relationship of jumping: the [MASK] is front of the [MASK].")

# transPhone.generateHexVectors()

for v in verbs:
    ger_v = getInflection(v, tag='VBG')[0].rstrip()
    # ger_v = v
    print(ger_v)
    # Define the sentence with two blanks
    relationships = ['is ' + phrase for phrase in ['inside', 'outside', 'in front of', 'behind', 'on the top of', 'under', 'adjacent to']]
    # relationships = ['is within', 'touches', 'crosses', 'overlaps']
    # sentences = ["In the spatial relation of " + ger_v + ", the [MASK] " + rel + " the [MASK]." for rel in relationships]
    sentences = ["Fill in the blanks to express a spatial relationship between two concepts using the 9 Chinese trigrams: '[MASK] is on [MASK] for stars'."]
    filled_s = []
    scores = []
    fillings_sets = []
    for s in sentences:
        result = fill_sentence(s)
        filled_s.append(result[0])
        scores.append(result[1])
        fillings_sets.append(result[2])

    print("Chosen:", filled_s[np.argmax(scores)], fillings_sets[np.argmax(scores)])
