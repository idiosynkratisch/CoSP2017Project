#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Create Topic objects and pickle them in dict.

Dict structure:         {'Topic ID': <Topic>}

@author: hanamollerkalpak
"""
from swda_time import *
import cPickle as pickle
from topic_segmenter_swda import *

# set corpus
corpus = CorpusReader('bnc_complete', 'bnc_complete/bnc-metadata.csv')

# segment all transcripts in corpus
wtf = []

seg_transcripts = []

for transcript in corpus.iter_transcripts(display_progress=True):
    try:
        seg_transcripts.append(SegmentedTranscript(transcript).segmented())
    except ValueError:
        wtf.append(transcript.conversation_no)
        continue
        
# assign numeral IDs to Topic objects and store in dict
topics = {}

topicID = 0

for seg_transcript in seg_transcripts:
    for topic in seg_transcript:
        ID = '%i' % topicID
        topics[ID] = topic
        topicID += 1

# write topics dict to file
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# switchboard
# save_object(topics, 'topics.pkl')       
        
#bnc 
save_object(topics, 'topics_bnc.pkl')

