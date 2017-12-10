#!/usr/bin/env python 

"""
Functions to convert dialogues from the BNC XML into transcripts and metadata
in a format compatible with swda.py

When run as a script it will automatically look for dialogues from the
demographically-sampled portion of the BNC in the local copy of BNC XML
and convert them, placing them in folders 'bnc' and 'bnc_pos', where
the latter contains transcripts including the CLAWS5-POS-tags provided
with BNC XML
"""
import os
from csv import DictWriter
from nltk.corpus.reader.bnc import BNCCorpusReader
from collections import defaultdict

corpus = BNCCorpusReader(root='BNC XML/Texts/', fileids=r'[A-K]/\w*/\w*\.xml')

def get_data(item, last_shifts=[]):
    """ Constructs text and pos-tag for the item supplied
        Arguments:
        item (xml.etree.Element): The item to be converted
        last_shifts (list of str): description of last <shift> events
        
        Returns:
        text (str): Text for item
        pos (str): Pos-tagged version of item
    """
    
    if item.tag in {'w', 'c'}:
        text = item.text
        try:
            pos = '{}/{} '.format(item.attrib['hw'], item.attrib['c5'])
        except KeyError:
            pos = '{}/{} '.format(item.text.rstrip(), item.attrib['c5'])
    elif item.tag == 'mw':
        #generate text and pos-tag recursively
        l = [get_data(it, last_shifts) for it in item]
        text = ''.join([i[0] for i in l])
        pos = '[{}]/{} '.format(''.join([i[1] for i in l]).rstrip(), item.attrib['c5'])
    elif item.tag in {'event', 'vocal'}:
        text = '<{}> '.format(item.attrib['desc'])
        pos = ''
    elif item.tag == 'align':
        text = ''
        pos = ''
    elif item.tag == 'gap':
        text = '<<REDACTED ({})>> '.format(item.attrib['desc'])
        pos = ''
    elif item.tag == 'pause':
        text = ''
        pos = ''
    elif item.tag == 'shift':
        if 'new' in item.attrib:
            text = '<{}> '.format(item.attrib['new'])
            last_shifts.append(item.attrib['new'])
        else:
            if last_shifts:
                text = '</{}> '.format(last_shifts.pop())
            else:
                text = ''
        pos = ''
    elif item.tag == 'unclear':
        if item.text == None:
            text = '(()) '
        else:
            text = '(({})) '.format(item.text)
        pos = ''
    elif item.tag == 'trunc':
        # generate text and pos-tag recursively
        l = [get_data(it, last_shifts) for it in item]
        # add '-' at the end of the truncated text
        text = ''.join([i[0] for i in l]).rstrip() + '- '
        # concatenate available pos-tags
        pos = ''.join([i[1] for i in l])
    else:
        print 'Unrecognized tag: {}'.format(item.tag)
        raise KeyboardInterrupt
    
    return text, pos

def write_transcript(conv, doc, directory, doPos=False):
    """ Writes a transcript of the conversation in csv-format and
        returns filename and ids of callers A and B
    
        Argument:
        conv (xml.etree.Element): The conversation to be transcribed.
        doc (str): The file containing the conversation
        directory (str): Directory to be written to
        pos (boolean): Transcribe BNC/CLAWS POS-tags?
    """
    filename = os.path.join(directory, conv.attrib['n'] + '.csv')
    header = ['swda_filename',       #original BNC XML file
              'ptb_basename',        #id of the recording
              'conversation_no',     #id of the div
              'transcript_index',    #index of subutterance (corresponds to <s>) in the transcript
              'act_tag',             #unused
              'caller',              #A (first speaker) or B (second speaker)
              'turn_index',          #index of utterance (corresponds to <u> or top-level <unclear>)
                                     #in the transcript
              'subutterance_index',  #index of subutterance within utterance (<s> within <u>)
              'text',                #text of subutterance (<s>)
              'pos',                 #POS-tags of subutterance
              'trees',               #field for trees to be added by the Stanford Parser
             ]
    
    n = conv.attrib['n']
    #try finding recording and setting id as attributes of the conversation
    try:
        recording, setting = conv.attrib['decls'].split()[:2]
    except KeyError:
        # if this fails try finding the corresponding recording and setting 'manually'
        xml = corpus.xml(doc)
        rec = xml.find(".//recording[@n='{}']".format(n))
        se = xml.find(".//setting[@n='{}']".format(n))
        if rec == None:
            print 'Failed to find recording ID, skipping'
            return None
        else:
            recording = rec.attrib['{http://www.w3.org/XML/1998/namespace}id']
        if se == None:
            print 'Failed to find setting ID, skipping'
            return None
        else:
            setting = se.attrib['{http://www.w3.org/XML/1998/namespace}id']
    
    with open(filename, 'w') as f:
        writer = DictWriter(f, header)
        writer.writeheader()
        
        # variables for ids of first and second speaker
        A = None
        B = None
        transcript_index = 0
        turn_index = 0
        prev_speaker = None
        
        for utt in conv.findall('*[@who]'):
            #set values that are the same for all utterances in a conversation
            d = {'swda_filename': doc,
                 'ptb_basename': recording,
                 'conversation_no': n}
            # find out who is speaking
            if utt.attrib['who'] == A:
                d['caller'] = 'A'
            elif utt.attrib['who'] == B:
                d['caller'] = 'B'
            elif A == None:
                A = utt.attrib['who']
                d['caller'] = 'A'
            elif B == None:
                B = utt.attrib['who']
                d['caller'] = 'B'
            else:
                print 'Something went wrong: Could not identify speaker'
                raise KeyboardInterrupt
            #if speaker has changed update and write turn_index, reset subutterance_index
            if prev_speaker != d['caller']:
                turn_index += 1
                subutterance_index = 1
            d['turn_index'] = turn_index
            #check if we have a top-level unclear element
            if utt.tag == 'unclear':
                #if so, write '(())' as text and continue
                d['transcript_index'] = transcript_index
                d['subutterance_index'] = subutterance_index
                d['text'] = '(())'
                writer.writerow(d)
                transcript_index += 1
                continue
            # intialize stack for shifts in vocal quality
            last_shifts = []
            for subutt in utt:
                if subutt.tag == 's':
                    # generate text and pos-tag recursively
                    l = [get_data(it, last_shifts) for it in subutt]
                    # concatenate text
                    text = ''.join([i[0] for i in l]).rstrip()
                    # concatenate available pos-tags
                    pos = ''.join([i[1] for i in l]).rstrip()
                else:
                    text, pos = get_data(subutt, last_shifts)
                d['transcript_index'] = transcript_index
                d['subutterance_index'] = subutterance_index
                d['text'] = text
                # throw away pos if pos-flag is not set
                # (TODO: prevent pos from even being computed if not necessary)
                if doPos:
                    d['pos'] = pos
                writer.writerow(d)
                transcript_index += 1
                subutterance_index += 1
            prev_speaker = d['caller']
    
    return n, A, B, recording, setting
    
    
def write_metadata(doc, ids, writer):
    """ Compiles metadata for conv and writes it using writer
    
        Argument:
        doc (xml.etree.Element): The XML-tree containing the conversation for which the metadata
                                 is to be written
        ids (iterable): Contains ids for conversation, caller A and B,
                        as well as recording and setting IDs
        writer (csv.DictWriter): DictWriter object to the metadata file.
    """
    
    # helper dicts to translate from BNC metadata to SWBD-like metadata
    sex = {'m': 'MALE', 'f': 'FEMALE', 'u': 'UNKNOWN'}
    # since exact birth dates are not always available in BNC, we choose the midpoint of each
    # age group and substract it from the date of recording to obtain an estimate, if possible.
    age_groups = {'Ag0': 10, 'Ag1': 20, 'Ag2': 30, 'Ag3': 40, 'Ag4': 52, 'Ag5': 75}
    dialect_areas = {'CAN': 'Canadian',
                     'NONE': 'UNK',
                     'XDE': 'German',
                     'XEA': 'East Anglian',
                     'XFR': 'French',
                     'XHC': 'Home Counties',
                     'XHM': 'Humberside',
                     'XIR': 'Irish',
                     'XIS': 'Indian subcontinent',
                     'XLC': 'Lancashire',
                     'XLO': 'London',
                     'XMC': 'Central Midlands',
                     'XMD': 'Merseyside',
                     'XME': 'North-east Midlands',
                     'XMI': 'Midlands',
                     'XMS': 'South Midlands',
                     'XMW': 'North-west Midlands',
                     'XNC': 'Central Northern England',
                     'XNE': 'North-east England',
                     'XNO': 'Nothern England',
                     'XOT': 'UNK',
                     'XSD': 'Scottish',
                     'XSL': 'Lower south-west England',
                     'XSS': 'Central south-west England',
                     'XSU': 'Upper south-west England',
                     'XUR': 'European',
                     'XUS': 'American (US)',
                     'XWA': 'Welsh',
                     'XWE': 'West Indian'}
    education = {'Ed1': 0, 'X':9}
    
    n, A, B, rec_id, set_id = ids
    
    #fill the rows that can already be filled
    d = {'conversation_no': n, 'from_caller_id': A, 'to_caller_id': B}
    
    recording = doc.find(".//recording[@n='{}']".format(n))
    date = None
    try:
        date = recording.attrib['date'].split('-')
        # format date in the swda way
        d['talk_day'] = recording.attrib['date'].replace('-','')[2:]
    except KeyError:
        # swda_time expects this to be set, so we set it to '010101' if undefined
        d['talk_day'] = '010101'
    
    try:
        d['length'] = recording.attrib['dur']
    except KeyError:
        # swda_times expects this to be set, so we set it to '0' if undefined
        d['length'] = 0
    
    setting = doc.find(".//setting[@n='{}']".format(n))
    try:
        d['topic_description'] = setting.find("activity").text
    except AttributeError:
        pass
    
    id_str = '@{http://www.w3.org/XML/1998/namespace}id'
    
    A = doc.find(".//person[{}='{}']".format(id_str, A))
    B = doc.find(".//person[{}='{}']".format(id_str, B))
    
    files = {'A': {}, 'B': {}}
    
    # try finding values for personal information, preferring the more explicit information
    for person in files:
        for att in ['sex', 'age', 'educ', 'dialect']:
            value = eval(person).find(att)
            if value:
                files[person][att] = value.text
            else:
                try:
                    files[person][att] = eval(person).attrib[att]
                except KeyError:
                    files[person][att] = ''
        if files[person]['age'] == '':
            try:
                ageGroup = eval(person).attrib['ageGroup']
                files[person]['age'] = age_groups[ageGroup]
            except KeyError:
                pass
        # compute birth_year if possible
        if files[person]['age'] != '' and date:
            files[person]['birth_year'] = int(date[0]) - int(files[person]['age'])
        else:
            # swda_time expects birth year to be set, so we set it to 1789 if unknown
            files[person]['birth_year'] = 1789
        dialect = files[person]['dialect']
        # convert dialect area if necessary
        if dialect in dialect_areas:
            files[person]['dialect'] = dialect_areas[dialect]
        # set dialect area to 'UNK' if unknown
        if dialect == '':
            files[person]['dialect'] = 'UNK'
        # convert education level if possible, else set to unknown (i.e. '9')
        educ = files[person]['educ']
        if educ in education:
            files[person]['educ'] = education['educ']
        else:
            files[person]['educ'] = 9
        # convert sex to swda format
        bnc_sex = files[person]['sex']
        if bnc_sex in sex:
            files[person]['sex'] = sex[bnc_sex]
        else:
            files[person]['sex'] = 'UNKNOWN'
            
    
    # use found personal data to fill corresponding field in the row
    d['from_caller_sex'] = files['A']['sex']
    d['from_caller_education'] = files['A']['educ']
    d['from_caller_birth_year'] = files['A']['birth_year']
    d['from_caller_dialect_area'] = files['A']['dialect']
    
    d['to_caller_sex'] = files['B']['sex']
    d['to_caller_education'] = files['B']['educ']
    d['to_caller_birth_year'] = files['B']['birth_year']
    d['to_caller_dialect_area'] = files['B']['dialect']
    
    # write compiled row to the metadata file
    writer.writerow(d)
            
    
def convert_bnc(dialogues, directory='bnc', doPos=False):
    """ Iterates over fileids and converts all contained conversations to csv-files
        writing metadata in the process
        
        Arguments:
        dialogues(dict of [str, list of str]):
            Dictionary indexed by relative paths of files containing
            conversations to be transcribed, with value a list of ids
            (<div:n>) of those conversations
        directory: Relative path of the directory to be written to
        doPos: Use POS-tags from BNC?
    """
    metadata_header = ['conversation_no',
                       'talk_day',
                       'length',
                       'topic_description',
                       'prompt',
                       'from_caller_sex',
                       'from_caller_education',
                       'from_caller_birth_year',
                       'from_caller_dialect_area',
                       'to_caller_sex',
                       'to_caller_education',
                       'to_caller_birth_year',
                       'to_caller_dialect_area',
                       'from_caller_id',
                       'to_caller_id'
                        ]
                        
    #create directory if it does not exist yet
    try:
        os.mkdir(directory)
    except OSError:
        pass
    
    metadata_filename = os.path.join(directory, 'bnc-metadata.csv')
    
    with open(metadata_filename, 'w') as metadata_file:
        meta_writer = DictWriter(metadata_file, metadata_header)
        meta_writer.writeheader()
        
        for doc in dialogues:
            doc_xml = corpus.xml(doc)
            for conv_id in dialogues[doc]:
                conv = doc_xml.find(".//div[@n='{}']".format(conv_id))
                ids = write_transcript(conv, doc, directory, doPos)
                write_metadata(doc_xml, ids, meta_writer)

if __name__ == '__main__':                
    # find files containing spoken text
    spoken = []

    for text in corpus.fileids():
        xml = corpus.xml(text)
        if xml.findall('stext') != []:
            spoken.append(text)
        
    # find subset of files containing spoken text that form the demographically sampled part
    # indicated by type 'CONVRSN'
    dem = []

    for doc in spoken:
        xml = corpus.xml(doc)
        for text in xml.findall('stext'):
            if text.attrib['type'] == 'CONVRSN':
                dem.append(doc)
            
    dialogues = defaultdict(list)

    #find files containing dialogues and compile a list of their indices per file
    for doc in dem:
        xml = corpus.xml(doc)
        for conv in xml.iter('div'):
            speakers = set()
            for utt in conv.iter('u'):
                speakers.add(utt.attrib['who'])
            if len(speakers) == 2:
                dialogues[doc].append(conv.attrib['n'])
            
    #clean up the Robert/Robert2 confusion (see roberts.txt for an explanation)
    dialogues['K/KD/KDT.xml'].remove('133401')
    dialogues['K/KD/KDT.xml'].remove('133501')
    dialogues['K/KD/KDT.xml'].remove('133502')
    dialogues['K/KD/KDT.xml'].remove('133602')

    #convert BNC without POS-tags to default folder bnc
    convert_bnc(dialogues)

    #convert BNC with included POS-tags to folder bnc_pos
    convert_bnc(dialogues, 'bnc_pos', doPos=True)