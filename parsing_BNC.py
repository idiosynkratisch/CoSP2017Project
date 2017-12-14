from swda_time import CorpusReader
import csv
import os
import re
from jpype import *

output_folder = "output_bnc"

try:
    os.mkdir(output_folder)
except OSError:
    pass


corpus = CorpusReader('bnc', 'bnc/bnc-metadata.csv')

#print tagged

#start the JVM
stanford_folder = "stanford-parser-full-2017-06-09"
startJVM(getDefaultJVMPath(),
         "-ea",
         "-mx2048m", 
         "-Djava.class.path={}".format(stanford_folder))
         
#import all needed Java classes
nlp = JPackage("edu").stanford.nlp
StringReader = java.io.StringReader
WhitespaceTokenizer = nlp.process.WhitespaceTokenizer
tokenizerFactory = WhitespaceTokenizer.newCoreLabelTokenizerFactory()

#intialize tagger
model_path = "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger"
tagger = nlp.tagger.maxent.MaxentTagger(model_path)

#load the parser
parser = nlp.parser.lexparser.LexicalizedParser.loadModel()

for trans in corpus.iter_transcripts(display_progress=True):
    parsed_trees = []
    pos_tags = []
    transcript = []
    paused = False
    for utterance in trans.utterances:
        if paused == True:
            index.append(utterance.transcript_index)
            if utterance.text.strip().startswith('--'):
                text = utterance.text.strip()[2:]
                unparsed_ut = unparsed_ut + text
            else:
                print 'oh no what happened, this corpus is shit'
                raise KeyboardInterrupt
            #unparsed_ut = str(utterance.text)
            if unparsed_ut.strip().endswith('--'):
                unparsed_ut = unparsed_ut.strip()
                unparsed_ut = unparsed_ut[:-2]
            else:
                paused = False
                transcript.append([trans.conversation_no, tuple(index), unparsed_ut])
        else:
            index = []
            index.append(utterance.transcript_index)
            if utterance.text.strip().endswith('--'):
                paused = True
                unparsed_ut = utterance.text.strip()
                unparsed_ut = unparsed_ut[:-2]
            else:
                unparsed_ut = utterance.text
                transcript.append([trans.conversation_no, tuple(index), unparsed_ut])

    #print transcript
    for line in transcript:
        pattern = re.compile('<.*?>')
        matches = re.findall(pattern, line[2])
        pattern = re.compile('\(\(.*?\)\)')
        matches += re.findall(pattern, line[2])
        for match in matches:
            line[2] = line[2].replace(match, '')
            
        unwanted = ["(", ")"]
        for item in unwanted:
            line[2] = line[2].replace(item, '')

        punctuation = ["'", "?", "!", ".", ",", ";", ":"]
        for item in punctuation:
            line[2] = line[2].replace(item, ' '+item)
        
        #tokenize and pos-tag the utterance
        text = tokenizerFactory.getTokenizer(StringReader(line[2])).tokenize()
        tagged_ut = tagger.tagSentence(text)
        line.append(tagged_ut.toString())
        
        #parse the tagged utterance
        parsed_ut = parser.parse(tagged_ut).toString()
        line.append(parsed_ut)

    #write to csv
    file_name = output_folder+'/'+str(trans.conversation_no)+'pos_tree.csv'
    with open(file_name, 'wb') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows(transcript)

#close the JVM again             
shutdownJVM()