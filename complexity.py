#!/usr/env/python

import os, commands
from collections import defaultdict
from nltk.tree import Tree
import numpy as np
from swda_time import CorpusReader
import re

# use swda as the default corpus
corpus = CorpusReader('swda', 'swda/swda-metadata.csv')

# list of names of functions to be used as measures for averaging
measures = ['depth', 'width', 'balanced', 'avdepth']

#dict for holding average measures per length
averages = dict([(measure, defaultdict(list)) for measure in measures])

# flag whether ranges have been computed
computed_ranges = False
# dict for the value ranges of depth and width
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
    
#------------------------------------------------------------
#implementation of Lu (2010)'s measures
  
#tregex patterns for subtrees of interest
#sentence (S)
s="'ROOT'"

#verb phrase (VP)
vp="'VP > S|SINV|SQ'"
vp_q="'MD|VBZ|VBP|VBD > (SQ !< VP)'"

#clause (C)
c="'S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]'"

#T-unit (T)
t="'S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]'"

#dependent clause (DC)
dc="'SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])'"

#complex T-unit (CT)
ct="'S|SBARQ|SINV|SQ [> ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]] << (SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]))'"

#coordinate phrase (CP)
cp="'ADJP|ADVP|NP|VP < CC'"

#complex nominal (CN)
cn1="'NP !> NP [<< JJ|POS|PP|S|VBG | << (NP $++ NP !$+ CC)]'"
cn2="'SBAR [<# WHNP | <# (IN < That|that|For|for) | <, S] & [$+ VP | > VP]'"
cn3="'S < (VP <# VBG|TO) $+ VP'"

#fragment clause
fc="'FRAG > ROOT !<< (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])'"

#fragment T-unit
ft="'FRAG > ROOT !<< (S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP])'"

#list of patterns for which we need the actual trees
matchlist=[c,t,fc,ft]
#list of patterns for which we only need the counts
countlist=[s,vp,dc,ct,cp,cn1,cn2,cn3,vp_q]

#list of measures 
lu_measure_list = ['MLC', 'MLT',
                   'C/S',
                   'C/T', 'CT/T', 'DC/C', 'DC/T',
                   'CP/C', 'CP/T', 'T/S',
                   'CN/C', 'CN/T', 'VP/T']
                   
tregex = "./L2SCA-2016-06-30/tregex.sh "
folder = "L2SCA-2016-06-30/"
 
def lus_measures(trees):
    """
    Takes an iterator over trees and computes the measures defined
    in Lu (2010) that make sense for single sentences.
    
    Returns a dict with the value for each measure indexed by the
    abbreviation from Lu (2010, Table 1)
    """
    #dict holding the matches
    matches=dict([(pattern, []) for pattern in matchlist])
    #dict holding the counts
    counts=dict([(pattern, 0) for pattern in countlist])
    
    #check if it has a root node, if not, add one:
    with open('trees', 'w') as temp:
        for tree in trees:
            if tree.label() != 'ROOT':
                tree = Tree('ROOT', [tree])
            temp.write(str(tree)+'\n')
    #retrieve the matching trees where we need them
    for pattern in matchlist:
        command = tregex +  "-s " + pattern + " " + "trees"
        output = commands.getoutput(command).split('\n')
        count_line = output[-1]
        number = re.compile('[0-9]+')
        #get the number of subtrees that have been found
        try:
            count = int(re.findall(number, count_line)[0])
        except IndexError:
            print 'Could not find number of matches'
            raise KeyboardInterrupt
        matches[pattern] = output[-(count+1):-1]
    #retrieve the counts for the other patterns
    for pattern in countlist:
        command = tregex + "-C -o " + pattern + " " + "trees"
        output = commands.getoutput(command).split('\n')
        count = float(output[-1])
        counts[pattern] = count
        
    #delete temp file again
    os.remove('trees')    
      
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
    results['C/S'] = counts[c]/counts[s]
    results['T/S'] = counts[t]/counts[s]
     
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
    
