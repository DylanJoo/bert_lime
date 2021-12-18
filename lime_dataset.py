"""
Function for perturbed examples, wrap-up the pipeline for pseduo-label prediction.
And make up the mini-batch huggingface procedure.
"""
# from dataclasses import dataclass
from datasets import Dataset
import multiprocessing
import collections
import time
import json
import numpy as np

class PerturbedDataset:
    """
    For the mini-batch perturbed and tokenized examples. 
    Original (instance) --> Perturbed (examples) --> Tokenized (dataset)

    Args:
        pairwise: default False, as only one input sentence (to-be-tokenized)
        tokenizer: the PretrainedTokenizer tailored for the model.
        device: 
    """

    def __init__(self, 
                 pairwise=False,
                 tokenizer=None, 
                 device='cpu'):
        # data
        self.pairwise = pairwise
        self.instances = collections.defaultdict(list)
        self.dataset = None
        self.examples = None

        self.tokenizer = tokenizer
        self.device = device

    def from_text(self):
        """
        Prefer all the into should be jsonl, and tokenized (by spacy)
        """
        pass

    def from_json(self, path_json):

        with open(path_json) as f:
            for line in f :
                example_dict = json.loads(line)
                if self.pairwise:
                    self.instances['wordsA'].append(example_dict['wordsA'])
                    self.instances['wordsB'].append(example_dict['wordsB'])
                else:
                    self.instances['words'].append(example_dict['words'])

        print(f"Loading json file from {path_json}, includes {len(self.instances)}")

    def perturbed_data_generation(self,
                                  num_samples=5000, 
                                  max_seq_length=128,
                                  distance_metric='cosine',
                                  perturbed_method='mask'):
        """ 
        Add the required perturbed data examples of each examples in the datast, 
        which indiactes that the size increase for 5000 times.

        (1) duplicate (initialize): copy the data into num_sample examples.
        (2) peturbation: (a) randomly select the perturbed amount (b) randonly select the perturbed token
        (3) tokenization: convert the perturbaed examples into huggingface tokens in tensors.

        Args:
            num_samples: the total number of self-constructed examples.
            perturbed_method: the neighborhood data generation process (for data-augmentation)
            - mask: Replace the selected token by [MASK].

        Returns:
            A tuple (perturbed, peturbed_data), where:
            perturbed: The N random pertrubing sentences.
            : The Bag-of-word representation of N perturbed sample.
        """
        # ****** 1  *****
        duplicated_instance = collections.defaultdict(list)
        for i in range(len(self.instances['wordsA'])):
            wordsA = self.instances['wordsA'][i]
            wordsB = self.instances['wordsB'][i]
            duplicated_instance['boolA'] += [[0] * len(wordsA)] * num_samples
            duplicated_instance['boolB'] += [[1] * len(wordsB)] * num_samples
            duplicated_instance['wordsA'] += [wordsA] * num_samples
            duplicated_instance['wordsB'] += [wordsB] * num_samples

        dataset = Dataset.from_dict(duplicated_instance)

        # ****** 2  *****
        def perturbation(examples):
            nB = len(examples['boolB'][0])
            try:
                perturbed_size_list = np.random.randint(1, nB, num_samples)
            except:
                perturbed_size_list = np.array([0] * num_samples)

            examples['sentA'] = [None] * num_samples
            examples['sentB'] = [None] * num_samples

            for i, num_perturbed in enumerate(perturbed_size_list):
                ## (a) ranomd picked the n tokens in sentence B by index
                if i != 0:
                    perturbed_indices = np.random.choice(
                            np.arange(nB),
                            num_perturbed,
                            replace=True
                    )
                else:
                    perturbed_indices = []

                # (b) perturb each examples
                for ii in range(nB):
                    if ii in perturbed_indices:
                        examples['wordsB'][i][ii] = "[MASK]"
                        examples['boolB'][i][ii] = 0

                examples['sentA'][i] = " ".join(examples['wordsA'][i])
                examples['sentB'][i] = " ".join(examples['wordsB'][i])

            return examples

        self.examples = dataset.map(
                function=perturbation,
                batched=True,
                batch_size=num_samples,
        )
        # Since the perturbing is random mask with fixed length in each instance, 
        # so it cannot multiprocess

        # ****** 3  *****
        def tokenization(examples):
            features = self.tokenizer(
                    examples['sentA'], examples['sentB'],
                    max_length=max_seq_length,
                    truncation=True,
                    padding='max_length'
            )
            return features


        # s = time.time()
        self.dataset = self.examples.remove_columns(['boolA', 'boolB', 'wordsA', 'wordsB']).map(
                function=tokenization,
                batched=True,
                keep_in_memory=True, 
        )
        # since the time spending on split and IO is longer than CPU time, 
        # multiprocessing has no benefit
        # print('TIME SPEND', time.time() - s)
        print(dataset)

    def get_bow_features(self):

        def merge_bools(examples):
            features = collections.defaultdict(list)
            for i in range(len(examples['boolA'])):
                features['bool'][i] = examples['boolA'][i] + examples['boolB'][i]

            return features

        bow_features = self.examples.remove_columns(['wordsA', 'wordsB']).map(
                function=merge_bools,
                batched=True,
                keep_in_memory=True
        )

        return bow_features


    def __getitem__(self, type_of_data):
        if type_of_data == 'original':
            return self.instances
        elif type_of_data == 'perturbed':
            return self.examples
        elif type_of_data == 'tokenized':
            return self.dataset
