"""
Functions for explaining text classifiers.
LimeBase: Lime's explainale model g.
LimeTextExplainer: Lime's locally perturbed explain algorithm.
"""
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import sklearn.metrics
import scipy as sp
from functools import partial
# from typing import Optional
import itertools
import json
import re
import collections
import numpy as np 
from sklearn.utils import check_random_state
from transformers import Trainer
import torch.nn.functional as f

from utils import batch_iter, flatten_listOflist
# from models import SimpleLinearRegression
from lime_dataset import PerturbedDataset


class LimeTextExplainer(object):

    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 token_selection_method='auto',
                 split_expression=r'\W+',
                 bow=True,
                 mask_string=None,
                 random_state=1234,
                 char_level=False, 
                 output_path='results.jsonl'):

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = random_state
        self.class_names = class_names
        self.importance = collections.defaultdict(list)
        self.token_selection_method = token_selection_method
        self.bow = bow
        self.char_level = char_level

        # to be build
        self.label_probs = np.empty((0, len(class_names)))
        self.text_instance = None
        self.output_path = output_path


    def explain_instance(self,
                         features,
                         prob_target,
                         labels,
                         max_num_features=None,
                         distance_metric='cosine',
                         model_regressor=None):
        """Function call for the raw string explaination pipeline.
        (1) load the explanation object
        (2) feature selection, return the considered feature (words)
        (3) Estimation, using the selected X and y

        Args:
            raw_string: the original text.
            labels: the original labels (which is used to infer the proba)
            max_num_features: specify the maximun number of features.
            num_samples: specify the number of perturbed samples.
        """
        def distances_fn(x):
            return sklearn.metrics.pairwise_distances(
                    x, x[:1, :], metric=distance_metric).ravel() * 100

        weights = self.kernel_fn(distances_fn(features))

        for label in labels:
            # ***** 2  *****
            idx_fs = self._feature_selection(
                    data=features,
                    labels=prob_target[:, label],
                    weights=weights,
                    num_features=max_num_features,
                    method='highest_weights', 
                    init_mask=None
            )
            Xs = features[:, idx_fs]

            # ***** 3  *****
            model_g = Ridge(
                    alpha=0, 
                    fit_intercept=False, 
                    random_state=self.random_state,
                    positive=True
            ) if model_regressor is None else model_regressor
            model_g.fit(Xs, prob_target[:, label], sample_weight=weights)
            ## esimation attributes
            fitness = model_g.score(Xs, prob_target[:, label], sample_weight=weights)
            y_pred = model_g.predict(Xs)
            coefficients = np.zeros(features.shape[1])
            coefficients[idx_fs] = model_g.coef_

            # ***** 4  *****
            self.importance[self.class_names[label]] = coefficients

    def _feature_selection(self,
                           data,
                           labels,
                           weights,
                           num_features=None,
                           method='highlight_weights',
                           init_mask=None):

        """
        Args:
            data: bag-of-words features with 0/1 of one instance (Original * 1 + Perurbed * (N-1))
            labels: order-sensitive classed indices
            weights: the distance (weighting criteria) of original and perturbed.
            init_mask: predefined mask for bow features, narrow the size of IV (X).
        Returns:
            used_features: the features which is subject to infleuncing the prediction.
        """
        init_mask = data[0, :].copy().astype(bool) if init_mask is None else init_mask.astype(bool)
        num_total_features = len(data[0, :])
        num_valid_features = sum(init_mask)
        assert num_valid_features == len(data[0, :]), 'Inconsistent between data and mask.'
        coefs = np.zeros(num_total_features)

        if method == 'highest_weights':
            """
            (a) Ridge regression without intercept. importance of each variables are estimated.
            [TODO] it's better to using regularization for the feature selection model.
            """
            current_mask = init_mask
            model = Ridge(alpha=0.01, random_state=self.random_state)
            model.fit(data[:, current_mask], labels, sample_weight=weights)
            coefs[current_mask] = model.coef_

            return np.flip(np.argsort(np.abs(coefs)))[:num_features]

        """
        (b) foward selection: select the features from nothing (empty set)
        (c) backward selection: select the features from all (full set)
        """
        if method == 'backward':
            consideration_mask = ~init_mask
            model = Ridge(alpha=0, random_state=self.random_state)

            for step in range(num_valid_features - num_features):
                selected = self.__get_new_feature(data, 
                                                  labels, 
                                                  weights, 
                                                  model, 
                                                  method, 
                                                  consideraion_mask)
                consideration_mask[selected] = True

            return np.flatnonzero(~consideration_mask)

        if method  == 'forward':
            consideration_mask = ~init_mask
            model = Ridge(alpha=0, random_state=self.random_state)

            for step in range(num_valid_features):
                selected = self.__get_new_feature(data, 
                                                  labels, 
                                                  weights, 
                                                  model, 
                                                  method, 
                                                  consideraion_mask)
                consideration_mask[selected] = True

            return np.flatnonzero(consideration_mask)

    def __get_new_feature(self,
                          Xs,
                          ys,
                          weights,
                          estimator,
                          direction,
                          mask):
        """
        Function of selecting a "best" features as the new features (to-be-add/to-be-removed)
            - In foward mode: add one feautre
            - In backward mode: remove one feature
        Args:
            mask: the mask contains all 'False' in the beginning.
        """
        scores = {}

        # Candidate feature index: the element which is not considered so far.
        for idx in np.flatnonzero(~mask):
            mask_new = mask.copy()
            mask_new[idx] = True 

            if direction == 'backward':
                mask_new = ~mask_new 

            estimator.fit(Xs[:, mask_new], ys, sample_weight=weights)
            scores[idx] = estimator.score(Xs[:, mask_new], ys, sample_weight=weights)

        return max(scores, key=lambda idx: scores[idx]) 
    
    def get_exp_list(self,
                     strings=None,
                     offset_items=None,
                     save_to_json=False):
        """
        [TODO] offset_items need to be added, if the further evaluation needs.
        """

        output_dict = {'word': strings}
        output_dict.update(self.importance)
        if save_to_json:
            with open(self.output_path, 'w') as f:
                f.write(json.dumps(output_dict) + '\n')

        return output_dict

