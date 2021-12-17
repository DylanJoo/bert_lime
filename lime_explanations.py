import collections 
import numpy as np


class Explanation:

    def __init__(self,
                 mode='classification',
                 class_names=None,
                 word_repr=None,
                 random_state=None):
        """The object for explanation, which wrapped up all the explanation and the features, 
        for the latter demostration or visualization.

        Args:
            mode: "classification" or "regression"
            class_names: list of class names (only used for classification)
            token_repr: the splitted representation of original sentneces.
            binary_repr: the binary representation for splitted tokens.
            seperate_repr: (1) single sentences (sentA: 1); (2) sentence pair (sentA: 0, sentB: 1)
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. 
        """
        self.random_state = random_state
        self.mode = mode
        self.class_names = class_names

        # The text information
        self.word_repr = None

        # The local explanation results
        for c in class_names:
            self.lime
        self.intercept = OrderedDict()
        self.coefficients = OrderedDict()
        self.scores = OrderedDict()
        self.prediction = {}

        # if mode == 'classification':
        #     self.class_names = class_names
        #     self.top_labels = None
        #     self.predict_proba = None
        # elif mode == 'regression':
        #     self.class_names = ['negative', 'positive']
        #     self.predicted_value = None
        #     self.min_value = 0.0
        #     self.max_value = 1.0
        #     self.dummy_label = 1
        # else:
        #     raise ValueError('Invalid explanation mode "{}"'.format(mode))

    def set_exp(self, tgt_lbl, exp_dict):
        """Function that process the explained coefficnet weights, 
        and apply them on the original splitted tokens."""

        assert tgt_lbl < len(self.class_names), \
                'Incorrect label, available classes: {}'.format("; ".join(self.class_names))

        targeted_name = self.class_names[tgt_lbl]

        self.intercept[targeted_name] = exp_dict['intercept']
        self.coefficients[targeted_name] = exp_dict['coefficients']
        self.scores[targeted_name] = exp_dict['scores']
        self.prediction[targeted_name] = exp_dict['prediction']

    def get_exp_list(self, topk):
        """Returns the explanation as a list."""
        ans = OrderedDict()
        for c in self.coefficients.keys():
            topk_indices = np.argsort(np.abs(self.coefficients[c]))[::-1][:topk]
            ans[c] = list(zip(self.token_repr[topk_indices], self.coefficients[c][topk_indices]))
         
        return ans
