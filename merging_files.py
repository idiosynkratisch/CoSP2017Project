import os, re, csv
from swda_time import CorpusReader

swbd_folder = "swda"
bnc_folder = "bnc"

swbd_metadata = "swda-metadata.csv"
bnc_metadata = "bnc-metadata.csv"

swbd = CorpusReader(swbd_folder, swbd_folder+"/"+swbd_metadata)

#figure out which conversation numbers are where for swda
filename = {}
for trans in swbd.iter_transcripts(display_progress=False):
    no = trans.conversation_no
    filename[no] = trans.swda_filename

swbd_trees_folder = "output"
bnc_trees_folder = "output_bnc"

swbd_output = "swda_complete"
bnc_output = "bnc_complete"

try:
    os.mkdir(swbd_output)
except OSError:
    pass
    
try:
    os.mkdir(bnc_output)
except OSError:
    pass

swbd_trees = os.listdir(swbd_trees_folder)
swbd_trees = [swbd_trees_folder+'/'+name for name in swbd_trees]
bnc_trees = os.listdir(bnc_trees_folder)
bnc_trees = [bnc_trees_folder+'/'+name for name in bnc_trees]

#copy the original switchboard files, which will be partially #overwritten, fixing filenames in the process

for no in filename:
    old = open(filename[no], 'r')
    new = open('{}/sw{}.csv'.format(swbd_output, no), 'w')
    new.writelines(old.readlines())
    old.close()
    new.close()

for path in swbd_trees:
    conv_no = int(re.findall("[0-9]+", path)[0])
    tree_file = open(path, 'rb')
    trans_file = open(filename[conv_no], 'rb')
    tree_reader = csv.reader(tree_file)
    trans_reader = csv.DictReader(trans_file)
    data = []
    for row in trans_reader:
        data.append(row)
    for row in tree_reader:
        for index in eval(row[0]):
            data[index]['trees'] = row[1]
    tree_file.close()
    trans_file.close()
    merged_file = open('{}/sw{}.csv'.format(swbd_output, conv_no), 'wb')
    wr = csv.DictWriter(merged_file, trans_reader.fieldnames)
    wr.writeheader()
    wr.writerows(data)
    merged_file.close()

#copy metadata file
original_metadata = open(swbd_folder+"/"+swbd_metadata, 'r')
new_metadata = open(swbd_output+"/"+swbd_metadata, 'w')
new_metadata.writelines(original_metadata.readlines())
original_metadata.close()
new_metadata.close()


for path in bnc_trees:
    conv_no = int(re.findall("[0-9]+", path)[0])
    trans_name = '{:0>6d}.csv'.format(conv_no)
    tree_file = open(path, 'rb')
    trans_file = open(bnc_folder+"/"+trans_name, 'rb')
    tree_reader = csv.reader(tree_file)
    trans_reader = csv.DictReader(trans_file)
    data = []
    for row in trans_reader:
        data.append(row)
    for row in tree_reader:
        for index in eval(row[1]):
            data[index]['conversation_no'] = conv_no
            data[index]['pos'] = row[3]
            data[index]['trees'] = row[4]
    tree_file.close()
    trans_file.close()
    merged_file = open('{}/bnc{}.csv'.format(bnc_output, conv_no), 'wb')
    wr = csv.DictWriter(merged_file, trans_reader.fieldnames)
    wr.writeheader()
    wr.writerows(data)
    merged_file.close()

#fix conversation numbers in metadata and copy the rest
original_metadata = open(bnc_folder+"/"+bnc_metadata, 'r')
reader = csv.DictReader(original_metadata)
metadata = []
for line in reader:
    line['conversation_no'] = int(line['conversation_no'])
    metadata.append(line)
original_metadata.close()
new_metadata = open(bnc_output+"/"+bnc_metadata, 'wb')
writer = csv.DictWriter(new_metadata, reader.fieldnames)
writer.writeheader()
writer.writerows(metadata)
new_metadata.close()