#!/usr/env/python

from collections import defaultdict
import numpy as np
from swda_time import CorpusReader

# use swda as the default corpus
corpus = CorpusReader('swda', 'swda/swda-metadata.csv')

# list of names of functions to be used as measures
measures = ['depth', 'width', 'balanced', 'avdepth', 'balanced2']

#dict for holding average measures per length
averages = dict([(measure, defaultdict(list)) for measure in measures])

computed_ranges = False
ran = {'depth': {}, 'width': {}}

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
    
def balanced2(tree, corpus=corpus):
    """
    Computes branching factor mapped to (0, 1) with depth mapped
    to (0, 1) so they are weighted equally
    """
    # check if value ranges have been computed, else do it
    
    global computed_ranges
    
    if not computed_ranges:
        values = dict([(measure, defaultdict(list))
                       for measure in ran])
        for utt in corpus.iter_utterances(display_progress=False):
            for tree in utt.trees:
                for measure in values:
                    values[measure][length(tree)].append(
                        float(eval(measure)(tree))
                    )
        for measure in values:
            for le in values[measure]:
                ran[measure][le] = (max(values[measure][le]),
                                    min(values[measure][le]))
        computed_ranges = True
    
    le = length(tree)
    #normalize depth
    if ran['depth'][le][0] == ran['depth'][le][1]:
    #if max and min a are equal, then all depths are average
        norm_depth = 0.5
    else:
    #compute range and normalize with it
        r_depth = ran['depth'][le][0] - ran['depth'][le][1]
        norm_depth = depth(tree)/r_depth - (ran['depth'][le][1]/r_depth)
    #do the same for width
    if ran['width'][le][0] == ran['width'][le][1]:
        norm_width = 0.5
    else:
        r_width = ran['width'][le][0] - ran['width'][le][1]
        norm_width = width(tree)/r_width - (ran['width'][le][1]/r_width)
        
    return norm_depth * norm_width
    
def _find_lengths(tree):
    """
    Helper function to go through a tree depth first and count the
    length of the branches
    """
    
    l = []
    
    #if we are at the end of a branch add 2 to the length
    if tree.height() == 2:
        return [2]
    #else compute the length of the branch starting at every
    #daughter node, add 1 and add them to the list
    else:
        for subtree in tree:
            l += map(lambda x: x+1, _find_lengths(subtree))
        return l
            
    
def avdepth(tree):
    """
    Computes the average length of the branches in tree
    """
    return np.mean(_find_lengths(tree))
    
    
def compute_averages(corpus=corpus):
    """
    Computes the averages per length for the measures listed
    """
    for utt in corpus.iter_utterances(display_progress=False):
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

def nbalanced2(tree, corpus=corpus):
    """
    Computes the normalized balanced measure
    of tree (using averages from corpus)
    """
    if not averages['balanced2']:
        compute_averages(corpus=corpus)
    return balanced2(tree)/averages['balanced2'][length(tree)]
    
def n_avdepth(tree, corpus=corpus):
    """
    Computes the normalized average depth of tree
    (using averages from corpus)
    """
    if not averages['avdepth']:
        compute_averages(corpus=corpus)
    return avdepth(tree)/averages['avdepth'][length(tree)]
    
