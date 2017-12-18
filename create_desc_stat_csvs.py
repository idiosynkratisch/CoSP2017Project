#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 12:15:50 2017

@author: hanamollerkalpak
"""
import numpy as np
import cPickle as pickle
from swda_time import *
from topic_segmenter_swda import *
from desc_stats import *
import csv

# set corpus
corpus = CorpusReader('bnc_complete', 'bnc_complete/bnc-metadata.csv')


# SWITCHBOARD
#with open('topics.pkl', 'rb') as input:
# BNC
"""with open('topics_bnc.pkl', 'rb') as input:
    topics = pickle.load(input)

with open('topics_roles_bnc.pkl', 'rb') as input:
    topics_roles = pickle.load(input)"""
    

measures = {'depth': depth, 'width': width, 'balanced': balanced, 
                 'balanced2': balanced2, 'avdepth': avdepth, 'ndepth': ndepth, 
                 'nwidth': nwidth, 'nbalanced': nbalanced, 'nbalanced2': nbalanced2, 'n_avdepth': n_avdepth}

all_info = get_all_info(['LEADER', 'FOLLOWER'], measures, topics_roles, topics)

download_dir = "bnc_desc_stats.csv"
f = open(download_dir, "w") 

columnTitleRow = ['transcriptID',
                  'topicID',
                  'speaker',
                  'uttID',
                  'within_trans_pos',
                  'within_topic_pos',
                  'depth',
                  'ndepth',
                  'width',
                  'nwidth',
                  'balanced',
                  'nbalanced',
                  'balanced2',
                  'nbalanced2',
                  'avdepth',
                  'n_avdepth'] 

writer = csv.DictWriter(f, columnTitleRow)
writer.writeheader()

for topic_ID in all_info.keys():
    data = {}
    topic = topics[topic_ID]
    data['transcriptID'] = topic.transcript.conversation_no
    data['topicID'] = topic_ID
    all_utts = [x for x in topic.transcript.utterances if 0 in range(len(x.trees))]
    sorted_all_utts = sorted(all_utts, key=lambda utterance: utterance.transcript_index)
    for role in all_info[topic_ID].keys():
        data['speaker'] = role
        for utt_ID in all_info[topic_ID][role].keys():
            data['uttID'] = utt_ID
            utt, data['within_topic_pos'], m_dict = all_info[topic_ID][role][utt_ID]
            for m in m_dict:
                data[m] = m_dict[m]
            data['within_trans_pos'] = sorted_all_utts.index(utt)
            
            #row = "%s, %s, %s, %s, %i, %i, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f \n " % (transcript_no, topic_ID, role, utt_ID, within_transcript_pos, pos,
            #                                                                          m_dict['depth'], m_dict['ndepth'],
            #                                                                          m_dict['width'], m_dict['nwidth'],
            #                                                                          m_dict['balanced'], m_dict['nbalanced'], m_dict['nbalanced2'],
            #                                                                          m_dict['balanced2'], m_dict['avdepth'], m_dict['n_avdepth'])
            writer.writerow(data)
            
f.close()