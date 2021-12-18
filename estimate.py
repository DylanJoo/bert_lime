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
        HfArgumentParser
)

@dataclass
class OurLimeAruguments:

    n_perturb: Optional[int] = field(default=1000)
    n_nonzero: Optional[int] = field(default=None)
    feature_selection: Optional[str] = field(default="highest_weight")
    result_file: Optional[str] = field(default=None)

@dataclass
class OurModelArguments:

    model_name_or_path: Optional[str] = field(default='textattack/bert-base-uncased-snli')
    model_base: Optional[str] = field(default='bert-base-uncased')
    config_name: Optional[str] = field(default='bert-base-uncased')
    tokenizer_name: Optional[str] = field(default='bert-base-uncased')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: Optional[bool] = field(default=True)
    model_revision: str = field(default="main")

@dataclass
class OurDataArguments:

    overwrite_cache: Optional[bool] = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    eval_file: Optional[str] = field(default="dev.jsonl")
    test_file: Optional[str] = field(default="test.jsonl")
    max_seq_length: Optional[int] = field(default=128)
    classes_names: List = field(default_factory=lambda: ['contradiction', 'neutral', 'entailment'])
    pairwise: Optional[bool] = field(default=True)


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

def main():
    # (1) Load the inferencing model (for prediction), which is the lime backbone model.
    parser = HfArgumentParser((
        OurModelArguments, OurDataArguments, OurTrainingArguments, OurLimeAruguments
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, lime_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, lime_args = parser.parse_args_into_dataclasses()

    ## config and tokenizer
    config_kwargs = {"output_hidden_states": True}
    tokenizer_kwargs = {}
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)

    ## models
    model_kwargs = {"cache_dir": model_args.cache_dir}

    ## Option 1: The finetined model 
    model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path
    )

    ## Option 2: Finetune my self (or used the pretrained model only (poor))
    # model = AutoModel.from_pretrained(
    #         model_args.model_name_or_path, 
    #         config=config,
    # )

    # (2) Prepare datasets
    dataset = PerturbedDataset(
            pairwise=data_args.pairwise,
            tokenizer=tokenizer,
    )
    dataset.from_json(
            path_json = data_args.test_file
    )
    ## perturbation generation and then tokenization
    dataset.perturbed_data_generation(
            num_samples=lime_args.n_perturb,
            max_seq_length=data_args.max_seq_length,
            distance_metric='cosine',
            perturbed_method='mask'
    )
    ## Data collator
    data_collator = DefaultDataCollator(
            return_tensors='pt'
    )

    # (3) Inferencing on all of them
    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator
    )
    # make the inferencing of logit on "classes"
    results = trainer.predict(
            test_dataset=dataset['tokenized'],
    )

    # (3) Lime Text explainer
    lime = LimeTextExplainer(
            kernel_width=25,
            kernel=None,
            class_names=data_args.classes_names,
            token_selection_method=lime_args.feature_selection,
            mask_string=None,
            random_state=1234,
            output_path=lime_args.result_file
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
                max_num_features=lime_args.n_nonzero,
                distance_metric='cosine',
                model_regressor=None,
        )
        # print(lime.importance)
        lime.get_exp_list(
                strings=dataset['perturbed']['wordsB'][s],
                offset_items=dataset['perturbed']['wordsA'][s],
                save_to_json=True if lime_args.result_file is not None else False
        )

if __name__ == '__main__':
    main()
