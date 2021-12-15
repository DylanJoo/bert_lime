
"""
Function for perturbed examples, wrap-up the pipeline for pseduo-label prediction.
And make up the mini-batch huggingface procedure.
"""
# from dataclasses import dataclass
from datasets import Dataset, DatasetDict
import multiprocessing
import collections
import json
import torch
import numpy as np

class PerturbedDataset:
    """For the mini-batch perturbed datset.,
    Args:
        data: the TextInstance to be transformed.
        tokenizer: the PretrainedTokenizer tailored for the model.
    [TODO]
        - Add the pertrubed function inside the multiprocessing pipeline.
        - Besides the huggingface dataset tokenizer, set the bow tokenizer
    """

    def __init__(self, 
                 pairwise=False,
                 text_instance=None, 
                 tokenizer=None, 
                 device='cpu',
                 bow=False):
        # data
        self.pairwise = pairwise
        self.dataset = None
        self.original = collections.defaultdict(list)
        self.pertrubed = None
        # self.add_from_instance(text_instance)  # add data_dict

        self.tokenizer = tokenizer
        self.bow = bow # indicate that's embedding layers
        self.device = device
        self.dataset = None


    def from_json(self, path_json):

        with open(path_json) as f:
            for line in f :
                example_dict = json.loads(line)
                if self.pairwise:
                    self.original['wordsA'].append(example_dict['wordsA'])
                    self.original['wordsB'].append(example_dict['wordsB'])
                else:
                    self.original['words'].append(example_dict['words'])

        print(f"Loading json file from {path_json}, includes {len(self.original)}")

    def perturbed_data_generation(self,
                                  num_samples=5000, 
                                  distance_metric='cosine',
                                  perturbed_method='mask'):
        """ 
        Add the required perturbed data examples of each examples in the datast, 
        which indiactes that the size increase for 5000 times.

        Args:
            num_samples: the total number of self-constructed examples.
            perturbed_method: the neighborhood data generation process (for data-augmentation)
            - mask: Replace the selected token by [MASK].

        Returns:
            A tuple (perturbed, peturbed_data), where:
            perturbed: The N random pertrubing sentences.
            perturbed_data: The Bag-of-word representation of N perturbed sample.
        """
        perturbed_data = collections.defaultdict(list)
        # (1) duplicate (initialize) the data into num_sample shards

        for i in range(len(self.original['wordsA'])):
            wordsA = self.original['wordsA'][i]
            wordsB = self.original['wordsB'][i]
            perturbed_data['boolA'] += [[0] * len(wordsA)] * num_samples
            perturbed_data['boolB'] += [[1] * len(wordsB)] * num_samples
            perturbed_data['wordsA'] += [wordsA] * num_samples
            perturbed_data['wordsB'] += [wordsB] * num_samples

        self.dataset = Dataset.from_dict(perturbed_data)


        # (2) peturbation
        def perturbation(examples):
            nB = len(examples['boolB'][0])
            ## select the perturbation size n
            try:
                perturbed_size_list = np.random.randint(1, nB, num_samples)
            except:
                perturbed_size_list = np.array([0] * num_samples)

            ## ranomd picked the n tokens in sentence B by index

            # features = {
            #         'boolA': collections.defaultdict(list), 
            #         'boolB': collections.defaultdict(list),
            #         'sentA': collections.defaultdict(list), 
            #         'sentB': collections.defaultdict(list),
            #         'wordsA': collections.defaultdict(list), 
            #         'wordsB': collections.defaultdict(list)
            # }

            for i, num_perturbed in enumerate(perturbed_size_list):
                if num_perturbed != 0:
                    perturbed_indices = np.random.choice(
                            np.arange(nB),
                            num_perturbed,
                            replace=True
                    )
                else:
                    perturbed_indices = []
                    
                # perturb
                for ii in range(nB):
                    if ii in perturbed_indices:
                        examples['wordsB'][i][ii] = "[MASK]"
                        examples['boolB'][i][ii] = 0

            return examples

        perturbed_dataset = self.dataset.map(
                function=perturbation,
                batched=True,
                batch_size=num_samples,
        )
        print("p2")

        # (3) tokenization
        def tokenization(examples):
            features = self.tokenizer(
                    examples['sentA'], examples['sentB'],
                    max_length=data_args.max_seq_length,
                    truncation=True,
                    padding=True
            )


        self.dataset = perturbed_dataset.map(
                function=tokenization,
                batched=True,
                batch_size=num_samples,
                remove_columns=['wordsA', 'wordsB'],
                num_proc=multiprocessing.cpu_count(),
        )

    def get_dataset(self):
        pass
        # preprocessed_dataset = self.dataset.map(
        #         function=prepare_feature,
        #         remove_columns=self.dataset.column_names,
        #         batched=True
        # )
        # if torch.cuda.is_available():
        #     preprocessed_dataset.set_format('torch', device='cuda:0')
        # else:
        #     preprocessed_dataset.set_format('torch', device='cpu')
        #
        # return preprocessed_dataset
