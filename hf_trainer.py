"""
The huggingface trainer for the following purpose:
(1a) Finetuning on the out-domain dataset (required)
(1b) Finetuning on the in-domain dataset. (optional)
(2) Inferencing the classification probabilites of perturbed samples.
"""
from transformers import Trainer
from datasets import load_dataset 
from models import BertForSequenceClassification, SimpleLinearRegression
import json


class HuggingfaceTrainer(Trainer):
    """
    [TODO] 
        - specify the evalaution strategy = 'step' or 'epoch'
        - specift the computeing metric function.

    Args:
        model: The hf's model
        tokenizer: The hf's tokenizer correponsing to the model
        args: training arguments (if None, use the default)
        train_datset: need to be set if do_train
    """

    def __post_init__(self):
        """
        Set the default model (f) setting: model, tokenizer
        """
        self._set_model(arg.model_name_or_path)
        self.tokenizer

        if self.args.do_train:
            # self.set_dataset()
            with open("config/training.config", 'a') as f:
                f.write(json.dumps(self.args))
                f.write("\n\n")

    def _set_dataset(self, name):
        """[TODO] the dataet should be preprocessed and tokenized"""
        datasets = load_dataset(name)
        self.train_dataset = datasets['train']
        self.eval_dataset = datasets['validation']

    # def train(self): the training process (follow the given training_args)
    # def evaluate(self):

class sklearnTrainer:

    def __init__(self):
        pass

    def foward(self):
        pass
    
    def predict(self):
        pass

