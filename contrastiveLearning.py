import torch
from transformers import BertForMaskedLM, BertTokenizer

# Load pre-trained BERT model and tokenizer
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the input sentence
input_sentence = "The cheetah is faster than the lion."

# Tokenize the input sentence
tokenized_sequence = tokenizer.encode(input_sentence, add_special_tokens=True, return_tensors='pt')

# Define the masked input sequence
masked_sequence = tokenized_sequence.clone()
mask_positions = torch.randint(len(tokenized_sequence), (2,))
for position in mask_positions:
    masked_sequence[position] = tokenizer.mask_token_id

# Generate the logits for the masked tokens in the masked input sequence
with torch.no_grad():
    outputs = model(masked_sequence)
    logits = outputs[0]

# Get the top predicted tokens for each mask in the masked input sequence
predicted_tokens = []
for position in mask_positions:
    predicted_token_ids = logits[position].argmax(-1).tolist()
    predicted_tokens.append(tokenizer.convert_ids_to_tokens([predicted_token_ids])[0])

# Print the predicted tokens for each mask in the masked input sequence
for i, position in enumerate(mask_positions):
    print("Top predicted token for mask '{}' in sentence '{}': {}".format(i+1, input_sentence, predicted_tokens[i]))
