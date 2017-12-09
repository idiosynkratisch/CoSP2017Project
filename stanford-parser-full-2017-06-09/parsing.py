from swda_time import CorpusReader
import nltk
from nltk.parse.stanford import *
from nltk.internals import find_jars_within_path
import csv
import os

total = 0
untagged = []
unannotated = []


corpus = CorpusReader('swda', 'swda/swda-metadata.csv')

for trans in corpus.iter_transcripts(display_progress=False):
    if filter(lambda x: x.pos != '', list(trans.utterances)) == []:
        untagged.append(trans.conversation_no)
    if filter(lambda x: x.trees != [], list(trans.utterances)) == []:
        unannotated.append(trans.conversation_no)
    total += 1

print len(untagged)
print len(unannotated)
print total
unannotated = set(unannotated) - set(untagged)

print(len(unannotated))
#print unannotated

parsed_trees = []
unannotated = list(unannotated)

#jar = 'stanford-parser.jar'

java_path = "/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/bin/java" # replace this
os.environ['JAVA_HOME'] = java_path

#"/Users/morwennahoeks/Documents/Github/CoSP2017Project/stanford-parser-full-2017-06-09/stanford-parser.jar"

parser=StanfordParser(path_to_jar='stanford-parser.jar', path_to_models_jar = 'stanford-parser-3.8.0-models.jar')
parser._classpath = tuple(find_jars_within_path("/Users/morwennahoeks/Documents/Github/CoSP2017Project/stanford-parser-full-2017-06-09"))

for trans in corpus.iter_transcripts(display_progress=True):
    for unparsed_convo in unannotated:
        if trans.conversation_no == unparsed_convo:
            parsed_trees = []
            for utterance in trans.utterances:
                utterance_no = utterance.transcript_index
                unparsed_ut = str(utterance.text)
                parsed_ut = parser.raw_parse(unparsed_ut)
                #print unparsed_ut, parsed_ut
                parsed_trees.append([utterance_no, list(parsed_ut)])

            # write to csv
            file_name = 'sw' + str(trans.conversation_no) + '.csv'
            with open(file_name, 'wb') as resultFile:
                wr = csv.writer(resultFile, dialect='excel')
                wr.writerows(parsed_trees)