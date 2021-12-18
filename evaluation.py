import collections
import argparse
import numpy as np
from spacy.lang.en import English
import json


# truth jsonl file
def load_from_jsonl(file_path):
    """
    Loading the groundtruth of the esnli development set, 
    Using the list of started (highlighted) tokens as the truth for each instance.
    """
    truth = collections.OrderedDict()
    sent = collections.OrderedDict()

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            truth[i] = json.loads(line)['keywordsB']
            sent[i] = json.loads(line)['sentA'] + " | " + json.loads(line)['sentB']

    return truth, sent

# prediction text files
def load_from_bert_lime(file_path, class_idx=0, prob_threshold=0, topk=None):
    """
    File type: dictionary file, e.g. .json, .jsonl
    """
    pred = collections.OrderedDict()
    punc = (lambda x: x in [",", ".", "?", "!"])

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            pred_per_example = []

            for j, (w, p) in enumerate(zip(data['word'], data[f'prob_{class_idx}'])):
                if punc(w) is False:
                    if p >= prob_threshold and punc(w) is False:
                        pred_per_example.append((w, p))

            pred[i] = [w for (w, p) in sorted(pred_per_example, key=lambda x: x[1])][:topk]
    return pred

def load_from_bert_seq_labeling(file_path, prob_threshold=0, sentA=True):
    """
    File type: dictionary file, e.g. .json, .jsonl
    Function for loading the predicted jsonl file, append the "selected" tokens, 
    which matched the requirements of "prob_threshold"
    """
    pred = collections.OrderedDict()
    punc = (lambda x: x in [",", ".", "?", "!"])

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            sentB = None
            pred[i] = []

            for j, (w, p) in enumerate(zip(data['word'], data['prob'])):
                if p == -1:
                    sentB = sentA if j == 0 else True
                elif p >= prob_threshold and sentB and punc(w) is False:
                    pred[i].append(w)
    return pred


def load_from_bert_span_detection(file_path):
    pass

def load_from_t5_mark_generation(file_path, show_negative=0):
    """
    File type: Raw text, e.g. .txt, .tsv
    """
    pred = collections.defaultdict(list)
    punc = (lambda x: x in [",", ".", "?", "!"])

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            hl = 0
            for tok in nlp(line.strip()):
                if tok.text == "*":
                    hl = 0 if hl else 1
                elif show_negative:
                    # pred[i] += [(tok.text, 1)] if hl else [(tok.text, 0)]
                    pred[i] += [tok.text] if hl else []
                else:
                    # pred[i] += [(tok.text, 1)] if hl else []
                    pred[i] += [tok.text] if hl else []
    return pred


def main(args):
    truth, strings = load_from_jsonl(args.path_truth_file)
    if args.output_type == 'bert-lime':
        pred = load_from_bert_lime(
                args.path_pred_file,
                class_idx=0,
                prob_threshold=0,
                topk=-1
        )
    elif args.output_type == 'bert-seq-labeling':
        pred = load_from_bert_seq_labeling(
                args.path_pred_file,
                prob_threshold=0.5, 
                sentA=False
        )
    elif args.output_type == 'bert-span-detection':
        pred = load_from_bert_span_detection(
                args.path_pred_file
        )
    elif args.output_type == 't5-marks-generation':
        pred = load_from_t5_mark_generation(
                args.path_pred_file,
                show_negative=0
        )
    else:
        print("Invalid type of highlight tasks")
        exit(0)

    assert len(truth) != len(pred), "Inconsisent sizes of truth and predictions"
    metrics = collections.defaultdict(list)

    for j, (truth_tokens, pred_tokens) in enumerate(zip(truth.values(), pred.values())):
        # [CONCERN] what if the tokens are redundant, revised if needed.
        print(f"Sentence pair: {strings[j]}\
                \n - Ground truth tokens: {truth_tokens}\
                \n - Highlighted tokens: {pred_tokens}")

        hits = set(truth_tokens) & set(pred_tokens)
        precision = (len(hits) / len(pred_tokens)) if len(pred_tokens) != 0 else 0
        recall = (len(hits) / len(truth_tokens)) if len(truth_tokens) != 0 else 0
        if precision + recall != 0:
            fscore = 2 * precision * recall / (precision + recall)
        else:
            fscore = 0

        if len(truth_tokens) != 0:
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(fscore)

    print("********************************\
            \nMean {:<9}: {:<5}\
            \nMean {:<9}: {:<5}\
            \nMean {:<9}: {:<5}\
            \nNum of evaluated samples: {}\
            \n********************************".format( 
                'precision', np.mean(metrics['precision']), 
                'recall', np.mean(metrics['recall']), 
                'f1-score', np.mean(metrics['f1']), j+1
            ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-truth", "--path_truth_file", type=str)
    parser.add_argument("-pred", "--path_pred_file", type=str)
    parser.add_argument("-hl_type", "--output_type", type=str)
    parser.add_argument("-eval_mode", "--evaluation_mode", type=str)
    # [TODO] Make the eval_mode flexible on the evaluation model,
    #   say highlightA & B or highlightB only
    args = parser.parse_args()
    nlp = English()

    main(args)




