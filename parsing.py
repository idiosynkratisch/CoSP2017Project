from swda_time import CorpusReader
import csv
import os
from jpype import *

output_folder = "output"

try:
    os.mkdir(output_folder)
except OSError:
    pass

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

unannotated = set(unannotated) - set(untagged)

#start the JVM
stanford_folder = "stanford-parser-full-2017-06-09"
startJVM(getDefaultJVMPath(),
         "-ea", 
         "-Djava.class.path={}".format(stanford_folder))

#import all needed Java classes
nlp = JPackage("edu").stanford.nlp
StringReader = java.io.StringReader
WhitespaceTokenizer = nlp.process.WhitespaceTokenizer
tokenizerFactory = WhitespaceTokenizer.newCoreLabelTokenizerFactory()
processor = nlp.process.WordToTaggedWordProcessor()

#load the parser
parser = nlp.parser.lexparser.LexicalizedParser.loadModel()

for trans in corpus.iter_transcripts(display_progress=True):
    if trans.conversation_no in unannotated:
        parsed_trees = []
        complete = {'A': False, 'B': False}
        indices = {'A': [], 'B': []}
        pos = {'A': '', 'B': ''}
        for utterance in trans.utterances:
            speaker = utterance.caller[-1]
            if utterance.act_tag == '+':
                complete[speaker] = False
            if complete[speaker] == True:
                unparsed_ut = pos[speaker]
                unparsed_ut = tokenizerFactory.getTokenizer(
                                  StringReader(unparsed_ut)
                                  ).tokenize()
                unparsed_ut = processor.process(unparsed_ut)
                parsed_ut = parser.parse(unparsed_ut).toString()
                #print unparsed_ut, parsed_ut
                parsed_trees.append([str(indices[speaker]),
                                    parsed_ut])
                pos[speaker] = ''
                indices[speaker] = []
            indices[speaker].append(utterance.transcript_index)
            #take out the continuation markers
            pos[speaker] += ' ' + utterance.pos.replace('--/:', '')
            complete[speaker] = True

        # write to csv
        file_name = 'sw' + str(trans.conversation_no) + '.csv'
        file_name = '{}/{}'.format(output_folder, file_name)
        with open(file_name, 'wb') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerows(parsed_trees)
  
#close the JVM again             
shutdownJVM()