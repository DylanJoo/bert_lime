"""
Functions and objects that for the text instances, 
which are ready to be explained.
"""
import collections
import spacy
import numpy as np
import sklearn.metrics
from utils import reformulation

nlp = spacy.load("en_core_web_sm")

class TextInstance:
    """Packing the text features and the related perturbed instances """

    def __init__(self,
                 raw_string=None,
                 split_expression=r'\W+',
                 perturbed_method='mask'):
        """Initializer for lime text instances.

        Args:
            raw_string: the "to-be-explained" sentence or sentence pairs(list).
            split_expression: Regex split by default.
        [TBC]: 
            The raw string need to be normalized (like single whitespaces)
        """

        self.raw = raw_string
        self.pairwise = False if len(raw_string) == 1 else True

        # The original sentence 
        self.original = collections.OrderedDict(
                {'sentA': self.raw[0], 'sentB': self.raw[1] if self.pairwise else []}
        )

        # The potential token to be replaced as perturbed data. (labeled True)
        self.split = collections.OrderedDict({'sentA': [], 'sentB': []})
        self.isfeature = collections.OrderedDict({'sentA': [], 'sentB': []})
        self.sent_sep = None
        self.num_features = collections.defaultdict(int)
        self._get_features()

        # The perturbing 
        # [TODO] perturbed_data is using binary vector. (invalid token included)
        self.perturbed_method = perturbed_method
        self.perturbed = collections.OrderedDict(
                {'sentA': [self.original['sentA']], 'sentB': [self.original['sentB']]}
        )
        self.perturbed_data = np.array([self.isfeature['sentA'] + 
                                        self.isfeature['sentB']]).astype(int)
        self.perturbed_distances = None

    def _get_features(self, ignore='sentA'):
        """Adopts the naive tokenization pipeline via spacy, (Keep the whitespace)"""

        for token in nlp(self.raw[0]):
            self.split['sentA'].append(token.text)
            self.isfeature['sentA'].append(not token.is_punct)
            if token.whitespace_:
                self.split['sentA'].append(" ")
                self.isfeature['sentA'].append(False)
            self.sent_sep = [1] * len(self.split['sentA'])

        if self.pairwise:  # Append the second sentence
            for token in nlp(self.raw[1]):
                self.split['sentB'].append(token.text)
                self.isfeature['sentB'].append(not token.is_punct)
                if token.whitespace_:
                    self.split['sentB'].append(" ")
                    self.isfeature['sentB'].append(False)
            self.sent_sep = [0] * len(self.split['sentA']) + [1] * len(self.split['sentB'])

        if (ignore == 'sentA') and (self.pairwise):
            self.isfeature['sentA'] = [False] * len(self.isfeature['sentA'])

        self.num_features['sentA'] = sum(self.isfeature['sentA'])
        self.num_features['sentB'] = sum(self.isfeature['sentB'])

    def get_num_words(self):
        """Returns the length of splitted sent/sent-pair."""
        return len(self.split['sentA'] + self.split['sentB'])

    def get_num_features(self):
        """Returns the length of valid features (tokens) of the sents."""
        return sum(self.num_features.values())

    def perturbed_data_generation(self,
                                  num_samples,
                                  distance_metric='cosine',
                                  perturbed_method='mask'):
        """Based on the given raw text, generate N perturbed examples.
        and the corresponding sparse representation.

        Args:
            num_samples: the total number of self-constructed examples.
            perturbed_method: the neighborhood data generation process (for data-augmentation)
            - mask: Replace the selected token by [MASK].

        Sub-functions:
            reformulation: returns the masked perturbing result of focal text.
            distances_fn: return the distances b/w samples and original on sparse representation

        Returns:
            A tuple (perturbed, peturbed_data), where: 
            perturbed: The N random pertrubing sentences.
            perturbed_data: The Bag-of-word representation of N perturbed sample.

        [TBC]
            In the paired scenario, I choose to perturb the sentB only (fixed the sentA).
        """

        def distances_fn(x):
            return sklearn.metrics.pairwise_distances(
                    x, x[:1, :], metric=distance_metric).ravel() * 100

        # perturbed 0: Copy the first row (original) to the text instance.
        self.perturbed_data = np.repeat(self.perturbed_data, repeats=num_samples + 1, axis=0)

        # perturbed 1: Generate "How many" token should be masked? of each example
        np.random.seed(1234) 
        num_perturbed_sample = np.random.randint(1, self.get_num_features(), num_samples)
        perturbed_idx_candidates = np.where(self.isfeature['sentA'] + self.isfeature['sentB'])[0]

        for i_example, num_perturbed in enumerate(num_perturbed_sample, start=1):
            # perturbed 2: select "which" tokens should be perturbed in each example
            perturbed_idx = np.random.choice(perturbed_idx_candidates, 
                                             num_perturbed, 
                                             replace=False)
            self.perturbed_data[i_example, perturbed_idx] = 0

            self.perturbed['sentA'].append(reformulation(self.split['sentA'], perturbed_idx))
            if self.pairwise:
                self.perturbed['sentB'].append(reformulation(self.split['sentB'], perturbed_idx, 
                                                        start=len(self.split['sentA'])))

        self.perturbed_distances = distances_fn(self.perturbed_data)
        # return self.pertrubed, self.perturbed_data, self.perturbed_distances
