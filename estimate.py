"""
Estimating function for sentence highlightin, 
which adopt the lime framework with bert model as the backbone.

Packages requirments:
    - hugginface
    - datasets
"""
import sys
import argparse
from dataclasses import dataclass, field

from typing import Optional, List, Union
import numpy as np
from scipy.special import softmax
from lime_text import LimeTextExplainer
from lime_dataset import PerturbedDataset
from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DefaultDataCollator,
        DataCollatorWithPadding,
        HfArgumentParser
)

@dataclass
class OurModelArguments:

    model_name_or_path: Optional[str] = field(default='bert-base-uncased')
    model_type: Optional[str] = field(default='bert-base-uncased')
    config_name: Optional[str] = field(default='bert-base-uncased')
    tokenizer_name: Optional[str] = field(default='bert-base-uncased')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: Optional[bool] = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: Optional[bool] = field(default=False)


@dataclass
class OurDataArguments:

    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: Optional[bool] = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    eval_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default="data.jsonl")
    max_seq_length: Optional[int] = field(default=128)
    classes_names: List = field(default_factory=lambda: ['contradiction', 'neutral', 'entailment'])


@dataclass
class OurTrainingArguments(TrainingArguments):

    output_dir: str = field(default='./models')
    do_train: Optional[bool] = field(default=False)
    do_eval: Optional[bool] = field(default=False)
    do_test: Optional[bool] = field(default=False)
    save_steps: int = field(default=1000)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=16)
    weight_decay: float = field(default=0.0)
    logging_dir: Optional[str] = field(default='./logs')
    warmup_steps: int = field(default=1000)
    resume_from_checkpiint: Optional[str] = field(default=None)

def main(lime_args):
    # (1) Load the inferencing model (for prediction), which is the lime backbone model.
    parser = HfArgumentParser((OurModelArguments, OurDataArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ## config and tokenizer
    config_kwargs = {"output_hidden_states": True}
    tokenizer_kwargs = {}
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)

    ## models
    model_kwargs = {"cache_dir": model_args.cache_dir}

    ## Option 1: The finetined model 
    model = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-snli')
    ## Option 2: Finetune my self (or used the pretrained model only (poor))
    # model = AutoModel.from_pretrained(
    #         model_args.model_name_or_path, 
    #         config=config,
    # )

    # (2) Prepare datasets
    dataset = PerturbedDataset(
            pairwise=True,
            tokenizer=tokenizer,
            device='cpu'
    )
    dataset.from_json(
            path_json = "data.jsonl"
    )
    ## perturbation generation and then tokenization
    dataset.perturbed_data_generation(num_samples=lime_args.n_perturb)
    ## Data collator
    # data_collator = DataCollatorWithPadding(
    #         padding=True,
    #         max_length=128,
    #         return_tensors='pt'
    # )
    data_collator = DefaultDataCollator(
            return_tensors='pt'
    )

    # (3) Inferencing on all of them
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            data_collator=data_collator
    )
    # make the inferencing of logit on "classes"
    results = trainer.predict(
            test_dataset=dataset['tokenized']
    )

    # (3) Lime Text explainer
    lime = LimeTextExplainer(
            kernel_width=25,
            kernel=None,
            verbose=False,
            class_names=data_args.classes_names,
            token_selection_method='auto',
            split_expression=r'\W+',
            bow=True,
            mask_string=None,
            random_state=1234,
            char_level=False,
    )

    ## Explain the instance one by one ...
    for s in range(0, len(dataset['perturbed']), lime_args.n_perturb):
        e = s + lime_args.n_perturb
        features = np.array(dataset['perturbed']['boolB'][s:e])
        probabilities = softmax(results.predictions[s:e], axis=-1)
        lime.explain_instance(
                features=features, # 2d array (N x #features)
                prob_target=probabilities, # 2d array (N x #classes)
                labels=(0,),
                max_num_features=None,
                distance_metric='cosine',
                model_regressor=None,
        )
        # print(lime.importance)
        t=lime.get_exp_list(
                strings=dataset['perturbed']['wordsB'][s],
                offset_items=dataset['perturbed']['wordsA'][s],
                save_to_json=False
        )
        print(dataset['perturbed']['wordsA'][s])
        print(t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-perturb_per_instance", "--n_perturb", type=int, default=5000)
    args = parser.parse_args()

    main(args)
