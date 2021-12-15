"""
Functions for the others programs, which are the small utilities.
- TextInstace
- LimeTextExplainer
"""
import numpy as np
import sklearn.metrics

def distances_fn(x):
    return sklearn.metrics.pairwise_distances(
            x, x[:1, :], metric=distance_metric).ravel() * 100

def reformulation(tokens, sub_index, start=0, sub_method='bert'):
    """For TextInstance.perturbed_data_generation().
    Usage: 
        Reformulate the original token feature with perturbbing, e.g. substitution
    """
    tokens = np.array(tokens, dtype=object)
    if start == 0:  # start from sentA(0), index from 0 to sentA's length.
        sub_index = sub_index[sub_index < len(tokens)]
    else:  # start from sentB, index from sentA'length to the last.
        sub_index = sub_index[sub_index >= start] - start
    tokens[sub_index] = '[MASK]'
    return "".join(tokens)

def batch_iter(iterable, size=1):
    """Batching the iterable (e.g. list, dataset)"""
    l = len(iterable)
    for ndx in range(0, l, size):
        yield iterable[ndx:min(ndx+size, l)]

def flatten_listOflist(lol, return_type='list'):
    if return_type == 'list':
        return [element for sublist in lol for element in sublist]
    elif return_type == 'np':
        return np.array([element for sublist in lol for element in sublist])

def assign_listTolist(lol, return_type='list'):
    pass
