from lime_dataset import PerturbedDataset
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

dataset = PerturbedDataset(
        pairwise = True,
        tokenizer = tokenizer,
        device = 'cuda:0',
        bow = True
)
dataset.from_json(
        "test.jsonl"
)
print("load")
dataset.perturbed_data_generation(num_samples=3)
print("processing")
print(dataset.dataset)
print(dataset.dataset['boolB'][:5])
print(dataset.dataset['wordsB'][:5])
print(dataset.dataset['sentB'][:5])
print(dataset.dataset['input_ids'][:5])
print(dataset.dataset)
