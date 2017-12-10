#!/usr/env/python

from collections import defaultdict
import numpy as np
from swda_time import CorpusReader

# use swda as the default corpus
corpus = CorpusReader('swda', 'swda/swda-metadata.csv')

# list of names of functions to be used as measures
measures = ['depth', 'width', 'balanced']

#dict for holding average measures per length
averages = dict([(measure, defaultdict(list)) for measure in measures])

def length(tree):
    """
    Computes the length of tree
    """
    return len(tree.flatten()) - 1
    
def depth(tree):
    """
    Computes the depth of tree
    """
    return tree.height()
    
def width(tree):
    """
    Computes the branching factor of tree
    """
    return np.mean([len(t) for t in tree.subtrees()])
    
def balanced(tree):
    """
    Computes branching factor multiplied with the depth of tree
    """
    return depth(tree) * width(tree)
    
def compute_averages(corpus=corpus):
    """
    Computes the averages per length for the measures listed
    """
    for utt in corpus.iter_utterances(display_progress=True):
        for tree in utt.trees:
            for measure in measures:
                averages[measure][length(tree)].append(
                                                eval(measure)(tree))
    for measure in measures:
        for le in averages[measure]:
            averages[measure][le] = np.mean(averages[measure][le])
        

def ndepth(tree, corpus=corpus):
    """
    Computes the normalized depth of tree (using averages from corpus)
    """
    if not averages['depth']:
        compute_averages(corpus=corpus)
    return depth(tree)/averages['depth'][length(tree)]
    
def nwidth(tree, corpus=corpus):
    """
    Computes the normalized width of tree (using averages from corpus)
    """
    if not averages['width']:
        compute_averages(corpus=corpus)
    return width(tree)/averages['width'][length(tree)]
    
def nbalanced(tree, corpus=corpus):
    """
    Computes the normalized balanced measure
    of tree (using averages from corpus)
    """
    if not averages['balanced']:
        compute_averages(corpus=corpus)
    return balanced(tree)/averages['balanced'][length(tree)]
    
