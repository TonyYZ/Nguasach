import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the available verbs and their associated basic components
verbs = {
    'speak': ['voice', 'words'],
    'articulate': ['speech', 'sounds'],
    'cook': ['food', 'heat', 'utensils'],
    'fly': ['body', 'movement', 'air'],
    'cry': ['tears', 'emotions']
}

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(verbs))
model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the training data
inputs = []
labels = []
for verb, components in verbs.items():
    for component in components:
        inputs.append(verb)
        labels.append(component)

encoded_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
labels = torch.tensor([list(verbs.keys()).index(verb) for verb in inputs])
dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], labels)

# Define the training loop
optimizer = AdamW(model.parameters(), lr=1e-5)
train_loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=16)
for epoch in range(10):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        for verb, components in verbs.items():
            inputs = [verb] * len(components)
            encoded_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
            labels = torch.tensor([list(verbs.keys()).index(verb)] * len(components)).to('cuda')
            outputs = model(encoded_inputs['input_ids'].to('cuda'),
                            attention_mask=encoded_inputs['attention_mask'].to('cuda'))
            predictions = torch.argmax(outputs.logits, dim=1)
            correct = (predictions == labels).sum().item()
            accuracy = correct / len(components)
            print(f'Verb: {verb}, Accuracy: {accuracy:.2f}')

encoded_inputs = tokenizer(["laugh"] * 2, padding=True, truncation=True, return_tensors='pt')
print(torch.argmax(model(encoded_inputs['input_ids'].to('cuda'),
                            attention_mask=encoded_inputs['attention_mask'].to('cuda')).logits, dim=1))