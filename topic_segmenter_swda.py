"""
    Tokenize swda transcript into topical sections using TextTiling.

    Converts Transcript object into SegmentedTranscript object (description below). 
                
    Method segmented() of SegmentedTranscript returns a chronologically sorted list of
    the topic segments of the original Transcript object, where topic segment == 
    Topic object; sentence == Sentence object. 

"""
#### initals #########

# imports
import nltk
nltk.download('stopwords')
nltk.download('brown')

# corpus reader
from swda_time import CorpusReader
corpus = CorpusReader('swda', 'swda/swda-metadata.csv')

# get Transcript class
from swda_time import Transcript

# topic segmenter
import nltk.tokenize.texttiling as texttiling

tt = texttiling.TextTilingTokenizer(demo_mode=False)

#################### 

class Sentence():
    """
    Utterance-like object with additional attributes for
        - lenght (number of words),
        - topic index ("within-topic position")
        
    To be added:
        - width
        - width * length
        - ev. other complexity measures
    """
    
    def __init__(self, utterance, segment):
        self.utterance = utterance
        self.segment = segment
        
        # within-topic position
        self.topic_index = self.segment.index(self.utterance)
        
        # list of words
        self.words = [word for word in self.utterance.pos_words()
                      if word not in (',', '.', '?')] # !!! exclude more markers?
        
        # lenght
        self.length = len(self.words)
        
        # manual inheritance of attributes of Utterance class
        self.caller = self.utterance.caller
        self.text = self.utterance.text
        self.pos = self.utterance.pos
        self.trees = self.utterance.trees

class Topic:
    """
    Topic episode with info about
        - lenght (number of sentences)
        - speaker roles (eg. leader() = 'A'/'B'/'NO ROLES')
        
    Method sentences() returns a chronologically ordered list
    of the Sentence objects of the topic episode.
    
    """
    
    def __init__(self, segment, segmented_utts):
        self.segment_no = segmented_utts.index(segment)
        self.segment = segment
        self.transcript = segmented_utts
        self.length = len(self.segment)
        
    def sentences(self):
        """
        Define sentences of the topic episode
        """
        sentences = []
        for utt in self.segment:
            sentences.append(Sentence(utt, self.segment))
        
        return sentences
    

    def leader(self):
        """
        Define the leader of the topic episode by Rule I and Rule II.
        
        Return 'A'/'B'
        """
        
        def utt_length_leader(n):
            """
            Define leader by Rule II: the speaker with the first 
            >= n word utterance in topic segment.
            
            """
            for i in range(self.length):
                if self.sentences()[i].length >= n:
                    leader = self.segment[i].caller
                    return leader
        
        # if topic shift ocurred: check if within a turn 
        if self.segment_no != 0:
            prev_topic = self.transcript[self.segment_no - 1]
            
            # if yes, assign leader role to speaker within 
            # whose turn the shift ocurred (Rule I)
            if self.sentences()[0].caller == prev_topic[len(prev_topic) - 1].caller:
                leader = self.sentences()[0].caller
                
                return leader
            
            # if no, assign leader role via Rule II
            else:
                leader = utt_length_leader(5)
                return leader
                
        # if first topic of transcript: Rule II
        else:
            leader = utt_length_leader(5)
            return leader
                    

    def follower(self):
        """
        Define the follower of the topic as the non-leader.
        If no leader, return 'NO ROLES'
        """
        if self.leader() == 'A':
            return 'B'
        elif self.leader() == 'B':
            return 'A'
        else:
            return 'NO ROLES'
    
        
    def leader_sentences(self):
        leader_sentences = [sent for sent in self.sentences() 
                            if sent.caller == self.leader()]
        
        return leader_sentences
    
    def follower_sentences(self):
        follower_sentences = [sent for sent in self.sentences()
                              if sent.caller == self.follower()]
        
        return follower_sentences
    

class SegmentedTranscript():
    """
    Tokenize swda transcript into topical sections using TextTiling.
                
    Method segmented() returns a chronologically sorted list of
    the topic segments of the transcript, where topic segment == 
    Topic object; sentence == Sentence object. 
    
    Procedure: Segment transcript by 
            1) tokenizing the transcript into text segments
            2) convert text segments into utterance segments
            3) convert utterance segments into Topic objects 
                (ie. Sentence segments)
    """
    
    def __init__(self, transcript):
        self.transcript = transcript
        self.utterances = [utt for utt in self.transcript.utterances]
        
    # tokenize transcript into text segments
    def texttile(self):
        utt_texts = []
        for utt in self.utterances:
            utt_texts.append(utt.text)
            
        segmented_text = tt.tokenize('\n\n\n\t'.join([utt for utt in utt_texts]))
        
        return segmented_text
        
    # count number of topics in transcript
    def topic_count(self):
        counter = len(self.texttile())
        return counter
        
    def segmented(self):
        """
        Convert text segments into utterance segments +
        convert utterance segments into Topic objects
        """
        
        segmented_utts = [] 
        topic_count = self.topic_count()
        
        # sort all topics in transcript chronologically
        sorted_transcript = sorted(self.utterances, key=lambda utterance:
                                   utterance.transcript_index)

        # populate segmented_utts
        for i in range(topic_count):
            segment = self.texttile()[i]
            number_of_sentences = len(segment.split('\n\n\n\t'))
            utterance_segment = sorted_transcript[:number_of_sentences]
            for x in utterance_segment:
                sorted_transcript.remove(x)
            segmented_utts.append(utterance_segment)
        
        # convert utterance segments to Topic objects
        segmented_transcript = []
        
        for segment in segmented_utts:
            segmented_transcript.append(Topic(segment, segmented_utts))
        
        return segmented_transcript
    
### test segmentation on individual transcript ###
"""trans = Transcript('swda_time/sw2005.csv', 'swda_time/swda-metadata-ext.csv')
seg_trans = SegmentedTranscript(trans)

+ code to extract the data of interest
"""

# segment all transcripts in corpus
"""
seg_transcripts = []
for transcript in corpus.iter_transcripts(display_progress=True):
    seg_transcripts.append(SegmentedTranscript(transcript))
"""





