#!/usr/env/python

import os, commands
from collections import defaultdict
from nltk.tree import Tree
import numpy as np
from swda_time import CorpusReader
import re
from jpype import *

def _remove_root(tree):
    if tree.label() == 'ROOT':
        return tree[0]
    else:
        return tree

def length(tree):
    """
    Computes the length of tree
    """
    return len(tree.flatten())

def depth(tree):
    """
    Computes the depth of tree
    """
    return _remove_root(tree).height()

def width(tree):
    """
    Computes the branching factor of tree
    """
    tree = _remove_root(tree)
    return np.mean([len(t) for t in tree.subtrees()])

def balanced(tree):
    """
    Computes branching factor multiplied with the depth of tree
    """
    tree = _remove_root(tree)
    return depth(tree) * width(tree)
    
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
    tree = _remove_root(tree)
    return np.mean(_find_lengths(tree))

def balanced2(tree, corpus, ranges=None):
    """
    Computes branching factor mapped to [1, 2] with depth mapped
    to [1, 2] so they are weighted equally
    
    The mapping to [1, 2] is obtained by feature scaling with the value ranges in ranges
    If these are not provided, they are computed
    """
    
    le = length(tree)
    #if ranges wer not given compute them for the length of tree
    if not ranges:
        ranges = {'depth': {}, 'width': {}}
        values = dict([(measure, defaultdict(list))
                       for measure in ranges])
        for utt in corpus.iter_utterances(display_progress=False):
            for tree in utt.trees:
                if length(tree) == le:
                    for measure in values:
                        values[measure][le].append(
                            float(eval(measure)(tree))
                            )
        for measure in values:
            for l in values[measure]:
                ranges[measure][l] = (max(values[measure][l]),
                                       min(values[measure][l]))

    #normalize depth
    if ranges['depth'][le][0] == ranges['depth'][le][1]:
    #if max and min a are equal, then all depths are average
        norm_depth = 1.5
    else:
    #compute range and normalize with it
        r_depth = ranges['depth'][le][0] - ranges['depth'][le][1]
        norm_depth = 1 + (depth(tree)/r_depth - (ranges['depth'][le][1]/r_depth))
    #do the same for width
    if ranges['width'][le][0] == ranges['width'][le][1]:
        norm_width = 1.5
    else:
        r_width = ranges['width'][le][0] - ranges['width'][le][1]
        norm_width = 1 + (width(tree)/r_width - (ranges['width'][le][1]/r_width))
    
    return norm_depth * norm_width


def compute_means(corpus,
                 measures=['depth', 'width', 'avdepth'],
                 ranges=None):
    """
    Computes the averages per length for measures, using ranges for balanced2
    """
    
    data = []
    means = dict([(measure, defaultdict(list)) for measure in measures])
    stdevs = dict([(measure, {}) for measure in measures])
    
    for trans in corpus.iter_transcripts(display_progress=True):
        complete = {('A',): False, ('B',): False, ('A', 'B'): True}
        last_trees = {}
        for utt in trans.utterances + ['EOC']:
            if not utt == 'EOC':
                speakers = (utt.caller[-1],)
            else:
            #if the transcript ended, write the last trees for both speakers
                speakers = ('A', 'B')
            if utt != 'EOC' and utt.act_tag == '+':
                complete[speakers] = False
            if complete[speakers] == True:
                for speaker in speakers:
                    for tree in last_trees[speaker]:
                        le = length(tree)
                        d = {'length':le}
                        for measure in measures:
                            if measure == 'balanced2':
                                if ranges:
                                    value = balanced2(tree, corpus, ranges=ranges)
                                else:
                                    value = balanced2(tree, corpus)
                            else:
                                value = eval(measure)(tree)
                            d[measure] = value
                            means[measure][le].append(value)
                        data.append([tree, d])
            if not utt == 'EOC':
                last_trees[speakers[0]] = utt.trees
                complete[speakers] = True
    for measure in measures:
        for le in means[measure]:
            stdevs[measure][le] = np.std(means[measure][le])
            means[measure][le] = np.mean(means[measure][le])
            
    return data, means, stdevs

def ndepth(tree, corpus):
    """
    Computes the normalized depth of tree (using averages from corpus)
    """
    averages = compute_averages(corpus, measures = ['depth'])
    return depth(tree)/averages['depth'][length(tree)]

def nwidth(tree, corpus):
    """
    Computes the normalized width of tree (using averages from corpus)
    """
    averages = compute_averages(corpus, measures = ['width'])
    return width(tree)/averages['width'][length(tree)]

def nbalanced(tree, corpus):
    """
    Computes the normalized balanced measure
    of tree (using averages from corpus)
    """
    averages = compute_averages(corpus, measures = ['balanced'])
    return balanced(tree)/averages['balanced'][length(tree)]

def nbalanced2(tree, corpus):
    """
    Computes the normalized balanced measure
    of tree (using averages from corpus)
    """
    averages = compute_averages(corpus, measures = ['balanced2'])
    return balanced2(tree, corpus)/averages['balanced2'][length(tree)]

def n_avdepth(tree, corpus):
    """
    Computes the normalized average depth of tree
    (using averages from corpus)
    """
    averages = compute_averages(corpus, measures = ['avdepth'])
    return avdepth(tree)/averages['avdepth'][length(tree)]
#------------------------------------------------------------
#implementation of Lu (2010)'s measures

#tregex patterns for subtrees of interest
#sentence (S)
s="ROOT"

#verb phrase (VP)
vp="VP > S|SINV|SQ"
vp_q="MD|VBZ|VBP|VBD > (SQ !< VP)"

#clause (C)
c="S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]"

#T-unit (T)
t="S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]"

#dependent clause (DC)
dc="SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])"

#complex T-unit (CT)
ct="S|SBARQ|SINV|SQ [> ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]] << (SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]))"

#coordinate phrase (CP)
cp="ADJP|ADVP|NP|VP < CC"

#complex nominal (CN)
cn1="NP !> NP [<< JJ|POS|PP|S|VBG | << (NP $++ NP !$+ CC)]"
cn2="SBAR [<# WHNP | <# (IN < That|that|For|for) | <, S] & [$+ VP | > VP]"
cn3="S < (VP <# VBG|TO) $+ VP"

#fragment clause
fc="FRAG > ROOT !<< (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])"

#fragment T-unit
ft="FRAG > ROOT !<< (S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP])"

#list of patterns for which we need the actual trees
matchlist=[c,t,fc,ft]
#list of patterns for which we only need the counts
countlist=[s,vp,dc,ct,cp,cn1,cn2,cn3,vp_q]

#dict for the compiled patterns
patterns = {}

#list of measures 
lu_measure_list = ['MLC', 'MLT',
                   'C/S',
                   'C/T', 'CT/T', 'DC/C', 'DC/T',
                   'CP/C', 'CP/T', 'T/S',
                   'CN/C', 'CN/T', 'VP/T']

#paths and objects for tregex
stanford_folder = "stanford-parser-full-2017-06-09"
nlp = None

def _init_tregex():
    """
    Initializes the JVM and compiles the tregex patterns.
    """
    global patterns, nlp
    #start the JVM
    startJVM(getDefaultJVMPath(),
             "-ea",
             "-mx2048m",
             "-Djava.class.path={}".format(stanford_folder))
         
    nlp = JPackage("edu").stanford.nlp
    TregexPattern = nlp.trees.tregex.TregexPattern
    for pattern in matchlist + countlist:
        patterns[pattern] = TregexPattern.compile(pattern)

def lus_measures(trees):
    """
    Takes an iterator over trees and computes the measures defined
    in Lu (2010) that make sense for single sentences.

    Returns a dict with the value for each measure indexed by the
    abbreviation from Lu (2010, Table 1)
    """

    #check if tregex has already been initiliazed, if not , dot it
    if patterns == {}:
        _init_tregex()

    #dict holding the matches
    matches=dict([(pattern, []) for pattern in matchlist])
    #dict holding the counts
    counts=dict([(pattern, 0.0) for pattern in countlist])

    #import edu.stanford.nlp.trees.Tree
    nlpTree = nlp.trees.Tree

    #retrieve the matching trees where we need them
    for tree in trees:
        if tree.label() != 'ROOT':
            tree = Tree('ROOT', [tree])
        #transform into Java object
        tree = nlpTree.valueOf(str(tree))
        for pattern in matchlist:
            matcher = patterns[pattern].matcher(tree)
            while matcher.findNextMatchingNode():
                match = matcher.getMatch().toString()
                matches[pattern].append(match)
        #retrieve the counts for the other patterns
        for pattern in countlist:
            matcher = patterns[pattern].matcher(tree)
            while matcher.findNextMatchingNode():
                counts[pattern] += 1.0
  
    #merge subcategories
    matches[c] = matches[c] + matches[fc]
    matches[t] = matches[t] + matches[ft]

    cn = 'cn'
    counts[cn] = counts[cn1] + counts[cn2] + counts[cn3]
    counts[vp] = counts[vp] + counts[vp_q]

    #add counts for clauses and t-units
    counts[c] = float(len(matches[c]))
    counts[t] = float(len(matches[t]))

    #see if there were any clauses and t-units
    if counts[c] == 0:
        noCs = True
    else:
        noCs = False
    
    if counts[t] == 0:
        noTs = True
    else:
        noTs = False

    #compute measures
    results = {}
    #compute mean length of clauses if possible
    c_lengths = [length(Tree.fromstring(tree)) for tree in matches[c]]
    if c_lengths !=[]:
        results['MLC'] = np.mean(c_lengths)
    else:
        results['MLC'] = None
    
    #compute mean length of T-units if possible
    t_lengths = [length(Tree.fromstring(tree)) for tree in matches[t]]
    if t_lengths !=[]:
        results['MLT'] = np.mean(t_lengths)
    else:
        results['MLT'] = None

    #compute the rest of the measures, if possible
    try:
        results['C/S'] = counts[c]/counts[s]
        results['T/S'] = counts[t]/counts[s]
    except ZeroDivisionError:
        results['C/S'] = None
        results['T/S'] = None
 
    if noTs:
        results['C/T'] = None
        results['CT/T'] = None
        results['DC/T'] = None
        results['CP/T'] = None
        results['CN/T'] = None
        results['VP/T'] = None
    else:
        results['C/T'] = counts[c]/counts[t]
        results['CT/T'] = counts[ct]/counts[t]
        results['DC/T'] = counts[dc]/counts[t]
        results['CP/T'] = counts[cp]/counts[t]
        results['CN/T'] = counts[cn]/counts[t]
        results['VP/T'] = counts[vp]/counts[t]
    
    if noCs:
        results['DC/C'] = None
        results['CP/C'] = None
        results['CN/C'] = None
    else:
        results['DC/C'] = counts[dc]/counts[c]
        results['CP/C'] = counts[cp]/counts[c]
        results['CN/C'] = counts[cn]/counts[c]
    
    return results
    

class ComplexityMeasures(object):
    """
    Class wrapper to easily work with complexity measures for multiple corpora
    Instantiate once for each corpus you are working with   
    """
    
    def __init__(self, corpus, normalized=['depth', 'width', 'avdepth']):
        #the corpus used
        self.corpus = corpus
        #the measure you want to be available in normalized versions
        self.normalized = set(normalized)

        #compute the value ranges for depth and width
        self.ranges = {'depth': {}, 'width': {}}
        values = dict([(measure, defaultdict(list))
                       for measure in self.ranges])
        for utt in self.corpus.iter_utterances(display_progress=False):
            for tree in utt.trees:
                le = length(tree)
                for measure in values:
                    values[measure][le].append(
                        float(eval(measure)(tree))
                    )
        print 'Ranges computed'
        for measure in values:
            for le in values[measure]:
                self.ranges[measure][le] = (max(values[measure][le]),
                                            min(values[measure][le]))
        #compute the averages for normalization                              
        self.data, self.averages, self.stdevs = compute_means(self.corpus,
                                                               measures=normalized,
                                                               ranges=self.ranges)
    
    
    def length(self, tree):
        return length(tree)
        
    def depth(self, tree):
        return depth(tree)
        
    def width(self, tree):
        return width(tree)
        
    def balanced(self, tree):
        return balanced(tree)
        
    def balanced2(self, tree):
        return balanced2(tree, self.corpus, self.ranges)
        
    def avdepth(self, tree):
        return avdepth(tree)
    
    def ndepth(self, tree):
        """
        Computes the normalized depth of tree (using averages from corpus)
        """
        return self.depth(tree)/self.averages['depth'][length(tree)]

    def nwidth(self, tree):
        """
        Computes the normalized width of tree (using averages from corpus)
        """
        return self.width(tree)/self.averages['width'][length(tree)]

    def nbalanced(self, tree):
        """
        Computes the normalized balanced measure
        of tree (using averages from corpus)
        """
        return self.balanced(tree)/self.averages['balanced'][length(tree)]

    def nbalanced2(self, tree):
        """
        Computes the normalized balanced measure
        of tree (using averages from corpus)
        """
        return self.balanced2(tree)/self.averages['balanced2'][length(tree)]

    def navdepth(self, tree):
        """
        Computes the normalized average depth of tree
        (using averages from corpus)
        """
        return self.avdepth(tree)/self.averages['avdepth'][length(tree)]
        
    def lus_measures(self, trees):
        return lus_measures(trees)

    def compute_normalized(self, measures=['depth', 'width', 'avdepth', 'balanced2']):
        nmeasures = []
        for measure in measures:
            if measure in self.normalized:
                nmeasures.append('n'+measure)
            else:
                nmeasures.append(measure)
        for measure in nmeasures:
            self.averages[measure] = defaultdict(list)
            self.stdevs[measure] = {}
        for point in self.data:
            for measure in nmeasures:
                value = eval("self.{}".format(measure))(point[0])
                point[1][measure] = value
                self.averages[measure][point[1]['length']].append(value)
        for measure in nmeasures:
            for le in self.averages[measure]:
                self.stdevs[measure][le] = np.std(self.averages[measure][le])
                self.averages[measure][le] = np.mean(self.averages[measure][le])
            
