import torch
from transformers import BertTokenizer, BertForMaskedLM, AdamW
import random
import re
import numpy as np
from lemminflect import getInflection
from datasets import load_dataset
import requests

skip_words = ['I', 'me', 'myself', 'you', 'it', 'she', 'he', 'we', 'they', 'them', 'one', 'another', 'this', 'that', 'being', "'", 'and', 'or', 'not', '"', '##s', '[CLS]', '-', ',', 'all', 'is', 'person', 'camera', 'man']
# skip_words = []
verbs = ['be', 'become', 'do', 'go', 'come', 'enter', 'exit', 'wait', 'stay', 'leave', 'return', 'think', 'consider', 'decide', 'choose', 'feel', 'see', 'hear', 'touch', 'smell', 'taste', 'say', 'know', 'learn', 'teach', 'repeat', 'continue', 'ask', 'answer', 'draw', 'write', 'read', 'eat', 'drink', 'have', 'hold', 'bring', 'get', 'give', 'send', 'put', 'leave', 'take', 'steal', 'hit', 'kick', 'rub', 'turn', 'push', 'pull', 'squeeze', 'wash', 'shower', 'wipe', 'scratch', 'cut', 'connect', 'stick', 'split', 'stab', 'laugh', 'smile', 'cry', 'bite', 'suck', 'swallow', 'spit', 'vomit', 'blow', 'breathe', 'cough', 'yawn', 'throw', 'catch', 'dig', 'tie', 'sew', 'count', 'wear', 'open', 'close', 'begin', 'finish', 'lose', 'look for', 'find', 'change', 'buy', 'pay', 'sell', 'try', 'make', 'create', 'use', 'break', 'ruin', 'kill', 'die', 'live', 'dream', 'sleep', 'wake up', 'lie', 'sit', 'squat', 'stoop', 'stand', 'walk', 'run', 'jump', 'swim', 'fly', 'move', 'stop', 'drive', 'follow', 'remember', 'forget', 'notice', 'ignore', 'add', 'remove', 'increase', 'decrease', 'rise', 'fall', 'descend', 'pass', 'sing', 'dance', 'play', 'float', 'flow', 'freeze', 'swell', 'grow', 'burn', 'cook', 'fry', 'boil', 'stew', 'steam', 'roast', 'bake', 'shine', 'melt', 'explode', 'love', 'like', 'want', 'need', 'hate', 'fear', 'kiss', 'penetrate', 'dash', 'shoot', 'fight', 'hunt', 'hurt', 'attack', 'protect', 'pile', 'store', 'insert', 'contain', 'include', 'inform', 'announce', 'sweep', 'rent', 'travel', 'record', 'escape', 'avoid', 'focus', 'bark', 'howl', 'call', 'scream', 'chirp', 'crow', 'shout', 'whisper', 'build', 'like', 'show', 'hang', 'weigh', 'measure', 'swing', 'shake', 'tremble', 'work', 'marry', 'pray', 'win', 'lose']
nouns = ['cooking', 'pot', 'food', 'stove', 'chimney', 'floor']


def flatten(l):
    return [item for sublist in l for item in sublist]

def import_double_hex():
    with open('SemanticsDoubleHexO.txt', encoding='utf-8') as f:
        content = f.readlines()
        double_hex_dataset = []
        for i in range(0, len(content), 3):
            hexCombo = content[i].split(' ')
            hexCombo[1] = hexCombo[1].rstrip()
            for j, w in enumerate(content[i + 2].split(' ')):
                if random.random() < 0.01 or j == 0:
                    w = w.rstrip()
                    double_hex_dataset.append(w + " is " + hexCombo[0] + " and " + hexCombo[1] + ".")
    return double_hex_dataset

firstExecution = False
if firstExecution:

    # Load pre-trained BERT model and tokenizer
    model = BertForMaskedLM.from_pretrained('bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    trigrams = [w.capitalize() for w in ['kun', 'gen', 'kan', 'xun', 'zhong', 'zhen', 'li', 'dui', 'qian']]
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

    # Define the domain-specific dataset
    dataset = ['The trigram ' + trigram + ' means ' + ', '.join(trigram_features[i]) + '.' for i, trigram in enumerate(trigrams)]
    dataset += ['The pot is on the stove when cooking.', 'the food is in the pot when cooking.']
    # dataset += import_double_hex()
    # big_dataset = load_dataset('ActivityNet_Captions.py', split='train')
    # big_dataset = load_dataset('conceptnet5.py')
    # print(big_dataset, big_dataset[0])
    #dataset += [*set(flatten([entry['en_captions'] for entry in big_dataset][:200]))]
    '''
    for i, v in enumerate(verbs):
        obj = requests.get('http://api.conceptnet.io/c/en/' + v).json()
        if i == 0:
            print(obj['edges'])
    '''
    with open('frameCorpus.txt', encoding='utf-8') as f:
        content = f.readlines()
        # print(content)
        for line in content:
            if random.random() < 0.15:
                dataset.append(line.rstrip())

    print(dataset, len(dataset))

    # Define the new words to be learned
    new_words = ['trigram'] + trigrams

    # Add new tokens to the tokenizer
    for word in new_words:
        tokenizer.add_tokens([word])

    # Resize the model's embedding matrix to accommodate the new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Tokenize the dataset
    tokenized_dataset = [tokenizer.encode(text, add_special_tokens=True) for text in dataset]
    # print(tokenized_dataset)

    # print(tokenizer.get_vocab())

    def create_masked_input(sequence, tokenizer, mask_prob=0.15):
        """
        Creates a masked version of the input sequence, where a random subset of tokens are replaced with the [MASK] token.
        Returns the masked sequence, the original sequence with the [MASK] tokens inserted, and a boolean mask indicating which
        tokens were masked.
        """
        # Create a boolean mask indicating which tokens to mask out
        is_masked = [random.random() < mask_prob if token != tokenizer.pad_token_id else False for token in sequence]
        # Apply the mask to the input sequence
        masked_sequence = [token if not is_masked[i] else tokenizer.mask_token_id for i, token in enumerate(sequence)]
        return masked_sequence, is_masked

    # Define the training parameters
    epochs = 1
    batch_size = 32
    learning_rate = 2e-5

    # Fine-tune the pre-trained BERT model on the domain-specific dataset
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        random.shuffle(tokenized_dataset)
        mean_loss = 0
        for i in range(0, len(tokenized_dataset), batch_size):
            print("Current batch:", i)
            batch = tokenized_dataset[i:i + batch_size]
            inputs = []
            labels = []
            for sequence in batch:
                # Mask out 15% of the tokens in the sequence at random
                masked_sequence, is_masked = create_masked_input(sequence, tokenizer)
                inputs.append(masked_sequence)
                labels.append(sequence)
            inputs = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in inputs], batch_first=True,
                                                     padding_value=tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in labels], batch_first=True,
                                                     padding_value=tokenizer.pad_token_id)
            optimizer.zero_grad()
            # print("inputs", inputs, "labels", labels)
            outputs = model(inputs, labels=labels)  # Compute the loss based on the masked_lm_labels
            loss = outputs.loss
            mean_loss += loss
            loss.backward()
            optimizer.step()
        print("Loss:", mean_loss / (len(tokenized_dataset) / batch_size))

    # Save the trained model
    model.save_pretrained('bert_model')
    tokenizer.save_pretrained('bert_tokenizer')
else:
    import_double_hex()
    model = BertForMaskedLM.from_pretrained('bert_model')
    tokenizer = BertTokenizer.from_pretrained('bert_tokenizer')

def replacenth(string, sub, wanted, n):
    where = [m.start() for m in re.finditer(sub, string)][n]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub.replace('\\', ''), wanted, 1)
    newString = before + after
    return newString


def fill_sentence(input_sentence, use_trigrams, reversed=False, skip_words_addition = []):
    # Use the fine-tuned BERT model to fill in masked trigrams
    tokenized_sequence = tokenizer.encode(input_sentence, add_special_tokens=True, return_tensors='pt')
    token_list = tokenizer.convert_ids_to_tokens(tokenized_sequence.tolist()[0])
    masked_sequence = tokenized_sequence.clone()[0]
    mask_positions = [i for i, x in enumerate(token_list) if x == "[MASK]"]
    # print(token_list, masked_sequence, mask_positions)
    trigram_start_ind = tokenizer.convert_tokens_to_ids('坤卦')
    weight_sum = 0
    fillings = []
    if reversed:
        enumerator = reversed(list(enumerate(mask_positions)))
    else:
        enumerator = enumerate(mask_positions)
    for i, position in enumerator:
        # print(i, position, token_list)
        with torch.no_grad():
            outputs = model(masked_sequence.unsqueeze(0))
        # print(logits[position][-15:], len(logits[position]))
        with torch.no_grad():
            if use_trigrams[i]:
                # print(outputs[0][0, position][-9:])
                predictions = outputs[0][0, position][-9:].topk(k=9)
            else:
                predictions = outputs[0][0, position][:-9].topk(k=10)
        # print("predictions: ", predictions, trigram_start_ind, predictions.indices[0] + trigram_start_ind)
        if use_trigrams[i]:
            predicted_tokens = [tokenizer.convert_ids_to_tokens(ind.item() + trigram_start_ind) for ind in
                                predictions.indices]
        else:
            predicted_tokens = [tokenizer.convert_ids_to_tokens(ind.item()) for ind in predictions.indices]
        predicted_weights = [tensor.item() for tensor in predictions.values]
        # print("Top predicted token for mask in sentence '{}': {}".format(input_sentence, list(zip(predicted_tokens, predicted_weights))))

        j = 0
        # print(result)
        for j in range(len(predicted_tokens)):
            if predicted_tokens[j] not in skip_words and predicted_tokens[j] not in fillings \
                    and predicted_tokens[j] not in skip_words_addition:
                break  # this word can be chosen
        choice = predicted_tokens[j]
        fillings.append(choice)
        weight_sum += predicted_weights[j]

        if reversed:
            replace_ind = i
        else:
            replace_ind = 0
        input_sentence = replacenth(input_sentence, '\[MASK\]', choice, replace_ind)
        if '[MASK]' not in input_sentence:
            print("The filled sentence: " + input_sentence + " (" +  str(weight_sum) + ").")
        tokenized_sequence = tokenizer.encode(input_sentence, add_special_tokens=True, return_tensors='pt')
        # token_list = tokenizer.convert_ids_to_tokens(tokenized_sequence.tolist()[0])
        masked_sequence = tokenized_sequence.clone()[0]
    return input_sentence, weight_sum, fillings

def batch_fill(use_trigrams):
    for v in verbs:
        inflected_v = getInflection(v, tag='VBG')[0].rstrip()
        print(inflected_v)
        # Define the sentence with two blanks
        relationships = ['is ' + phrase for phrase in
                         ['inside', 'outside', 'in front of', 'behind', 'on the top of', 'under', 'adjacent to']]
        # relationships = ['is within', 'touches', 'crosses', 'overlaps']
        # sentences = ["In the spatial relation of " + inflected_v + ", the [MASK] " + rel + " the [MASK]." for rel in relationships]
        tasks = ['The process of ' + inflected_v + " involves [MASK], [MASK], [MASK]."]
        #tasks = ["The meaning of '" + inflected_v + "' is consisted of the trigram [MASK] and the trigram [MASK]."]
        # tasks = ["In the process of " + inflected_v + ", the [MASK] " + rel + " the [MASK]." for rel in relationships]
        filled_sentences = []
        scores = []
        fillings_lists = []
        for task in tasks:
            result = fill_sentence(task, use_trigrams=use_trigrams, skip_words_addition=[inflected_v, v])
            filled_sentences.append(result[0])
            scores.append(result[1])
            fillings_lists.append(result[2])

        print("Chosen:", filled_sentences[np.argmax(scores)], fillings_lists[np.argmax(scores)])


#fill_sentence(
#    "the action of cooking's spatial relationship: the [MASK] is on the top of the [MASK].", use_trigrams=False)
batch_fill(use_trigrams=[False, False, False])
