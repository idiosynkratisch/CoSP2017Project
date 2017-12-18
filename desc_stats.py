#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Functions for gathering descriptive statistics.

Intended for use with structures like 
    topic           {'topic ID' : <Topic>}   
    topics_roles     {'topic ID' : {'SPEAKER ROLE' : 
                    [role-specific sentences in topic]}}
    
Created on Thu Dec 14 16:46:39 2017

@author: hanamollerkalpak

"""
import numpy as np
import cPickle as pickle
from topic_segmenter_swda import *
import csv
from complexity import *


# restore dict with Topic objects 
# structure: {'[numeral topic ID]' : <Topic>}

# SWITCHBOARD
#with open('topics.pkl', 'rb') as input:
# BNC
#with open('topics_bnc.pkl', 'rb') as input:
    #topics = pickle.load(input)
    
# restore dict with roles and utterance objects
# structure: {'[numeral topic ID]' : {'[SPEAKER ROLE]' : [role sentences of topic]}}  
    
# SWITCHBOARD
#with open('topics_roles_sw.pkl', 'rb') as input:
# BNC
   
#with open('topics_roles_bnc.pkl', 'rb') as input:
    #topics_roles = pickle.load(input)


# dicts for complexity measures
nonnormalized = {'depth': depth, 'width': width, 'balanced': balanced, 
                 'balanced2': balanced2, 'avdepth': avdepth}
normalized = {'ndepth': ndepth, 'nwidth': nwidth, 'nbalanced': nbalanced,
              'nbalanced2': nbalanced2, 'n_avdepth': n_avdepth}

# list of speaker roles
roles = ['LEADER', 'FOLLOWER']


# compute average number of topics per convo
def topics_per_convo(corpus):
    topics_per_convo = []
    wtf = []
    
            
    for transcript in corpus.iter_transcripts(display_progress=True):
        try:
            topic_count = SegmentedTranscript(transcript).topic_count()
            topics_per_convo.append(topic_count)
        except ValueError:
            wtf.append(transcript.conversation_no)
            continue
    sd = np.std(topics_per_convo)
    mean_topics_per_convo = np.mean(topics_per_convo)
    
    return mean_topics_per_convo, sd


# compute average number of utterances per topic segment
def sentences_per_topic(topics):
    sentences_per_topic = []
    
    for key in topics.keys():
        topic = topics[key]
        sentences_per_topic.append(topic.length)
        
    sd = np.std(sentences_per_topic)
    mean_sentences = np.mean(sentences_per_topic)
    
    return mean_sentences, sd


# compute average complexity of sentences for leaders / followers
def roles_averages(roles, measures, topics_roles):
    
    role_complexities = {role: {measure: [] for measure in measures.keys()} for role in roles}
    
    wtf = [] # collect non-tree sentences
    
    for topic_ID in topics_roles.keys():
        for role in roles:
            for utt in topics_roles[topic_ID][role]:
                if 0 in range(len(utt.trees)):
                    for measure in measures.keys():
                        f = measures[measure]
                        role_complexities[role][measure].append(f(utt.trees[0]))
                else:
                    wtf.append((utt, topic_ID))
    
    mean_role_complexities = {measure: {role: np.mean(role_complexities[role][measure])for role in roles} for measure in measures.keys()}
    
    return mean_role_complexities, wtf


def get_all_info(roles, measures, topics_roles, topics):

    all_the_info = {topic_ID: {role: {} for role in roles} for topic_ID in topics.keys()}
    
    for topic_ID in topics_roles.keys():
        topic = topics[topic_ID]
        topic_utts = [x for x in topic.segment if 0 in range(len(x.trees))]
        for role in roles:
            utts = [x for x in topic_utts if x in topics_roles[topic_ID][role]]
            for utt in utts:
                within_topic_position = topic_utts.index(utt)
                utt_ID = topic_ID + '_' + str(within_topic_position)
                measure_dict = {}
                for measure in measures.keys():
                    f = measures[measure]
                    complexity = f(utt.trees[0])
                    measure_dict[measure] = complexity
                        
                all_the_info[topic_ID][role][utt_ID] = (utt, within_topic_position, measure_dict)
                    
    return all_the_info



# find most/least complex sentence(s)
def find_extremes(all_the_info, measures):
    extremes = {measure: {'UPPER': (), 'LOWER': ()} for measure in measures}
 
    helper = {measure: {} for measure in measures}
    
    for topic_ID in topics_roles.keys():
        for role in roles:
            for measure in measures:
                utt = all_the_info[topic_ID][role][measure][0]
                complexity = all_the_info[topic_ID][role][measure][2]
                helper[measure] = {(utt,): complexity}
    
    all_complexities = {measure: sorted([x for x in helper[measure].values()]) for measure in measures}
    
    for measure in measures:
        upper = all_complexities[measure][len(all_complexities[measure]) - 1]
        lower = all_complexities[measure][0]
        
        upper_utts = []
        for key in helper[measure].keys():
            if helper[measure][key] == upper:
                upper_utts.append(key[0])
        
        lower_utts = []
        for key in helper[measure].keys():
            if helper[measure][key] == lower:
                lower_utts.append(key[0])
                
        extremes[measure]['UPPER'] = (upper, upper_utts)
        extremes[measure]['LOWER'] = (lower, lower_utts)
        
    return extremes
