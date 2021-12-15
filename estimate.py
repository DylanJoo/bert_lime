"""
Estimating function for sentence highlightin, 
which adopt the lime framework with bert model as the backbone.

Models: 
    (f) Bert 
    (g) Simple linear regression

Packages requirments:
    - hugginface
    - datasets
"""

import argparse
from lime_text import LimeBase, LimeTextExplainer

def main():
    """
    (1)
    (2)
    """

    # 1. Load the inferencing model (for prediction), which is the lime backbone model.
    # [TODO] Besides load from pretrained/finetuned model, using the finetuned process for it.
    parser = HfArgumentParser((OurModelArguments, OurDataArguments, OurTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config_kwargs = {"output_hidden_stats": True}
    tokenizer_kwargs = {}
    model_kwargs = {"cache_dir"L model_args.cache_dir}

    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    model = AutoModel.from_pretrained(
            model_args.model_name_or_path, 
            config=config,
            model_args=model_args
    )

    # 2. Add a lime Text explainer
    lime = LimeTextExplainer(
            kernel_width=25,
            class_names = data_args.classes_names
    )

    lime.set_trainer(model_args=model_args, training_args=training_args)





# Binary sentiment classification
SENT_A='This was the worst restaurant I have ever had the misfortune of eating at.'

lime_001 = LimeTextExplainer(class_names = ['negative', 'positive'])
lime_001.set_trainer(model_args=model_args, training_args=training_args)
lime_001.explaination = lime_001.explain_instance(
        raw_string = [SENT_A], 
        labels=(0, 1), 
        max_num_features=None,
        num_samples=5000
)

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

parser = argparse.ArgumentParser()
parser.add_argument("-sentA", "--sentence_text_A", type=str)
parser.add_argument("-sentB", "--sentence_text_B", type=str)
args = parser.parse_args()
