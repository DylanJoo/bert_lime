from lime_text import *
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
from transformers import (
        TrainingArguments,
        HfArgumentParser
)

@dataclass
class lime_TrainingArguments(TrainingArguments):
    eval_transfer: bool = field(default=False)
    # metadata={"help": "Evaluate transfer task dev sets (in validation)."}

@dataclass
class lime_ModelArguments:
    # metadata={"help": "The model checkpoint for weights initialization." "Don't set if you want to train a model from scratch."}
    # metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    # metadata={"help": "Pretrained config name or path if not the same as model_name"}
    # metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    # metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"}
    # metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    # metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    # metadata={"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script " "with private models)."}
    """SimCSE's arguments"""
    # metadata={"help": "Temperature for softmax."}
    # metadata={"help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last).}"
    # metadata={"help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."}
    # metadata={"help": "Whether to use MLM auxiliary objective."}
    # metadata={"help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."}
    # metadata={"help": "Use MLP only during training"}

    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    # temp: float = field(default=0.05)
    # pooler_type: str = field(default="cls")
    # hard_negative_weight: float = field(default=0)
    # do_mlm: bool = field(default=False)
    # mlm_weight: float = field(default=0.1)
    # mlp_only_train: bool = field(default=False)


parser = HfArgumentParser((lime_ModelArguments, lime_TrainingArguments))
model_args, training_args = parser.parse_args_into_dataclasses()

# Binary sentiment classification
# SENT_A='This was the worst restaurant I have ever had the misfortune of eating at.'
#
# lime_001 = LimeTextExplainer(class_names = ['negative', 'positive'])
# lime_001.set_trainer(model_args=model_args, training_args=training_args)
# lime_001.explaination = lime_001.explain_instance(
#         raw_string = [SENT_A], 
#         labels=(0, 1), 
#         max_num_features=None,
#         num_samples=5000
# )

# Sentence-pair
# SENT_A='A black race car starts up in front of a crowd of people.'
# SENT_B='A man is driving down a lonely road.'
#
SENT_A='As of September 24, 2016 ,' + \
      ' the Company had approximately 116,000 full-time equivalent employees.'
SENT_B='As of September 30, 2017 ,' + \
      ' the Company had approximately 123,000 full-time equivalent employees.'

lime_002 = LimeTextExplainer(class_names = ['entailment', 'neutral', 'contradiction'])
lime_002.set_trainer(model_args=model_args, training_args=training_args)
lime_002.explaination = lime_002.explain_instance(
        raw_string = [SENT_A, SENT_B], 
        labels=(0, 1, 2), 
        max_num_features=None,
        num_samples=5000
)
#

# Deprecated: No need the data training arugment, since we are not using the new task for finetuning
# @dataclass
# class DataTrainingArguments:
#     dataset_name: Optional[str] = field(default=None)
#     dataset_config_name: Optional[str] = field(default=None)
#     overwrite_cache: bool = field(default=False)
#     validation_split_percentage: Optional[int] = field(default=5)
#     preprocessing_num_workers: Optional[int] = field(default=None)
#     # metadata={"help": "The name of the dataset to use (via the datasets library)."}
#     # metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
#     # metadata={"help": "Overwrite the cached training and evaluation sets"}
#     # metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"
#     # metadata={"help": "The number of processes to use for the preprocessing."},
#
#     train_file: Optional[str] = field(default=None)
#     max_seq_length: Optional[int] = field(default=32)
#     pad_to_max_length: bool = field(default=False)
#     mlm_probability: float = field(default=0.15)
#     # metadata={"help": "The training data file (.txt or .csv)."}
#     # metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer " "than this will be truncated."}
#     # metadata={"help": "Whether to pad all samples to `max_seq_length`, If False, will pad the samples dynamically when batching to the maximum length in the batch."}
#     # metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
#
#     def __post_init__(self):
#         if self.dataset_name is None and self.train_file is None and self.validation_file is None:
#             raise ValueError("Need either a dataset name or a training/validation file.")
#         else:
#             if self.train_file is not None:
#                 extension = self.train_file.split(".")[-1]
#                 assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."

