#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Create and pickle a topics_roles dict

structure:          {'topic ID' : {'SPEAKER ROLE' : 
                    [role-specific sentences in topic]}}
    
Created on Sat Dec 16 16:16:22 2017

@author: hanamollerkalpak
"""
import cPickle as pickle
from swda_time import *

# set corpus
corpus = CorpusReader('bnc_complete', 'bnc_complete/bnc-metadata.csv')

# restore dict with Topic objects 
# BNC
with open('topics_bnc.pkl', 'rb') as input:
#Switchboard
# with open('topics.pkl', 'rb') as input:
    topics = pickle.load(input)
    
# function for finding topic leader
def find_leader(topic):
    """
    Define the leader of the topic episode by Rule I and Rule II.
    
    Return 'A'/'B'
    """
    
    topic_utts = topic.segment
    
    def utt_length_leader(n):
        """
        Define leader by Rule II: the speaker with the first 
        >= n word utterance in topic segment.
        
        """
        for i in range(len(topic_utts)):
            utt_length = [word for word in topic_utts[i].pos_words() if word not in (',', '.', '?')]  
            if utt_length >= n:
                leader = topic_utts[i].caller
                
                return leader
        
    # if topic shift ocurred: check if within a turn 
    if topic.segment_no != 0 and topic_utts != []:
        prev_topic = topic.seg_utts[topic.segment_no - 1]

        # if yes, assign leader role to speaker within 
        # whose turn the shift ocurred (Rule I)
        if topic_utts[0].caller == prev_topic[len(prev_topic) - 1].caller:
            leader = topic_utts[0].caller
                    
            return leader
                
        # if no, assign leader role via Rule II
        else:
            leader = utt_length_leader(5)
            return leader
                    
    # if first topic of transcript: Rule II
    else:
        leader = utt_length_leader(5)
        return leader
                               

topics_roles = {}

for topicID in topics.keys():
    topic = topics[topicID]
    topic_utts = topic.segment
    leader = find_leader(topic)
    if leader == 'A':
        follower = 'B'
    elif leader == 'B':
        follower = 'A'
    else:
        follower = 'NO ROLES'
    
    if follower != 'NO ROLES':
        leader_utts = [x for x in topic_utts if x.caller == leader]
        follower_utts = [x for x in topic_utts if x.caller == follower]
        topics_roles[topicID] = {'ALL': topic_utts, 'LEADER': leader_utts, 'FOLLOWER': follower_utts}

# write topics_roles dict to file
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# for switchboard
#save_object(topics_roles, 'topics_roles_sw.pkl')

# for BNC
save_object(topics_roles, 'topics_roles_bnc.pkl')
