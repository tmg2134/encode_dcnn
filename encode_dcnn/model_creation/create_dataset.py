import os
import argparse
import subprocess
import random

import numpy as np
import pysam
import h5py
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json
import encode_dcnn.tools.bedtools_operations as bto

# One hot encode sequence for CNN architecture
OHE_SEQ = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0]
}

def one_hot_encode_sequence(seq):
    return np.array([OHE_SEQ[s] for s in seq.upper()])

def extract_sequences(full_seq, subseq_len=500, window=0):
    if window > 0:
        if len(full_seq) % window != 0:
            return np.array([False])
        if subseq_len > len(full_seq):
            return np.array([False])

        return np.array([
            one_hot_encode_sequence(full_seq[i:i + subseq_len]) 
            for i in range(0, len(full_seq) - subseq_len, window)
        ])
    
    # if subseq_len != 500:
    #     return np.array([False])
    
    return np.array([one_hot_encode_sequence(full_seq)])

parser = argparse.ArgumentParser()
parser.add_argument(
    '--peaks', 
    required=True, 
    help='Bed file of regions that are Enhancers for model to be trained on'
)
# parser.add_argument('--notEnhancers', default=False)
parser.add_argument('--bedtools-path', default='bedtools')
parser.add_argument(
    '--expand-to',
    type=int,
    default=500,
    help='How far to expand all regions too in given bed file of enhancers.\n'+
         '\tAll regions need to be the same length if no window size given'
)
parser.add_argument('--input-feature-size', type=int, default=50)
parser.add_argument('--window-size', type=int, default=5)
parser.add_argument(
    '--subseq-len',
    type=int,
    default=500,
    help='length of each sub-sequence. Default is 500',
)
parser.add_argument(
    '--reference-fasta',
    help='Path to genome fasta',
    required=True
)
parser.add_argument(
    '--genome-sizes',
    help='Genome sizes file (BED format)',
    required=True
)
parser.add_argument(
    '--negative-input-ratio',
    type=float,
    default=1.0,
    help='Ratio of negative regions to create relative to given regions')
parser.add_argument(
    '--blacklist',
    default='blacklist/all_peaks_combined.bed',
    help='bed file of regions to not include in selection of "non-enhancers"'+
    '\nA good idea is to use a bed file of known enhancer regions',
)
parser.add_argument('--output', 
    default='dataset.h5', 
    help="Output name of dataset"
)
parser.add_argument(
    '--test-set',
    nargs='*',
    default=[],
    help="Chromosome number/letter to leave for the test set. IE: 7 9")
parser.add_argument(
    '--valid-set',
    nargs='*',
    default=[],
    help="Chromosome number/letter to leave for the valid set. IE: 3 5")
args = vars(parser.parse_args())

window_size = args['window_size']
subseq_len = args['subseq_len']

EXPAND = int(args['expand_to'] / 2)
test_set = [] 
valid_set = []

# I don't remember what this is for
# if args['notEnhancers']:
#     notEnhancers_regions_file = open(args['notEnhancers'])
#     notEnhancers = []
#     for i in notEnhancers_regions_file:
#         notEnhancers.append(i.strip())
# else:
#     notEnhancers = False

# Set up storing generated data
for i in args['test_set']:
    test_set.append('chr{}'.format(str(i)))
for i in args['valid_set']:
    valid_set.append('chr{}'.format(str(i)))
if len(test_set) > 0:
    X_t, y_t = list(), list()
if len(valid_set) > 0:
    X_v, y_v = list(), list()

# Prepare input data
my_region_set = bto.get_comp(
    args['peaks'],
    args['blacklist'],
    EXPAND,
    args['genome_sizes'],
    args['bedtools_path']
)

ref = pysam.FastaFile(args['reference_fasta'])
positive_seqs = []
# Generate input data sequences
# [1, 0] = enhancer region
# [0, 1] = not enhancer region

print('Generating input sequences')
X, y, input_i = list(), list(), 1
for record in my_region_set['enhancer_peaks']:
    input_i += 1
    if input_i % 9000 == 0:
        print('Generated {} data points'.format(input_i))
    chrom, start, stop = record.strip().split('\t')[:3]
    next_input_seq = ref.fetch(chrom, int(start), int(stop))
    input_seqs = extract_sequences(
        next_input_seq,
        subseq_len=subseq_len,
        window=window_size
    )
    # Add to correct set depending on if chromsomes being dropped or not
    # and if we are in that chromosome
    if chrom in test_set:
        X_t.extend(input_seqs)
        region_string = ':'.join([chrom, start, stop]) + '.{}'
        y_t.extend(
            [[1, 0, region_string.format(i)] 
            for i in range(len(input_seqs))]
        )
    elif chrom in valid_set:
        X_v.extend(input_seqs)
        region_string = ':'.join([chrom, start, stop]) + '.{}'
        y_v.extend(
            [[1, 0, region_string.format(i)] 
            for i in range(len(input_seqs))]
        )
    else:
        X.extend(input_seqs)
        # Format as "[region_string].[kmer_id]"
        region_string = ':'.join([chrom, start, stop]) + '.{}'
        y.extend(
            [[1, 0, region_string.format(i)] 
            for i in range(len(input_seqs))]
        )

    # positive_seqs.append(next_input_seq)

if len(test_set) > 0:
    num_pos_X_t = len(X_t)
if len(valid_set) > 0:
    num_pos_X_v = len(X_v)

j = 0
negative_seqs = []
print('Generating complement regions')
while j < int(input_i * args['negative_input_ratio'] - 1):
    if j+1 % 9000 == 0:
        print('Generated {} complement data points'.format(j))
    
    # can organize more clearly and create two seperate functions
    if notEnhancers == False:
        rand_region = random.randint(
            0, len(my_region_set['complement_regions'])-1
        )
        chrom, start, stop = my_region_set['complement_regions'][rand_region]
        if int(stop) - int(start) < args['expand_to']:
            del my_region_set['complement_regions'][rand_region]
            continue
    else:
        chrom, start, stop = notEnhancers[j]

    midpoint = random.randint(int(start) + EXPAND, int(stop) - EXPAND)
    next_input_seq = ref.fetch(chrom, midpoint - EXPAND, midpoint + EXPAND)
    if len(next_input_seq) < EXPAND:
        if notEnhancers:
            j += 1
        continue
    input_seqs = extract_sequences(
        next_input_seq,
        subseq_len=subseq_len,
        window=window_size
    )
    # Add correct amount of found regions to different test sets
    if len(test_set) > 0:
        if j < num_pos_X_t:
            X_t.extend(input_seqs)
            y_t.extend([[0, 1, ''] for i in range(len(input_seqs))])
        elif len(valid_set) > 0: 
            if j < num_pos_X_t + num_pos_X_v:
                X_v.extend(input_seqs)
                y_v.extend([[0, 1, ''] for i in range(len(input_seqs))])
            else:
                X.extend(input_seqs)
                y.extend([[0, 1, ''] for i in range(len(input_seqs))])
    elif len(valid_set) > 0: 
        if j < num_pos_X_v:
            X_v.extend(input_seqs)
            y_v.extend([[0, 1, ''] for i in range(len(input_seqs))])
        else:
            X.extend(input_seqs)
            y.extend([[0, 1, ''] for i in range(len(input_seqs))])
    else:
        X.extend(input_seqs)
        y.extend([[0, 1, ''] for i in range(len(input_seqs))])

    j += 1 # Keep track of generated sequences

print('Split into train-validation-test around (80/10/10) sets')
if len(test_set) > 0 and len(valid_set) > 0:
    X_train = np.array(X)
    y_train = np.array(y)
elif len(test_set) > 0:
    X_train, X_valid, y_train, y_valid = 
        train_test_split(np.array(X), np.array(y), test_size=0.12)
elif len(valid_set) > 0:
    X_train, X_test, y_train, y_test = 
        train_test_split(np.array(X), np.array(y), test_size=0.12)
else:
    X_train, X_interim, y_train, y_interim = 
        train_test_split(np.array(X), np.array(y), test_size=0.2)
    X_test, X_valid, y_test, y_valid = 
        train_test_split(X_interim, y_interim, test_size=0.5)

# Create region mapping
y_with_regions = np.array(y)
sequence_id_map = defaultdict(list)
for i, label in enumerate(y):
    if label[2]:
        region_string, seq_id = label[2].split('.')
        sequence_id_map['{}'.format(i)] = label[2]

# Remove region string and kmer ID from y labels
write_X = np.array(X)
write_y = np.array(y)
if len(test_set) > 0:
    X_test = np.array(X_t)
    y_test = np.array(y_t)
if len(valid_set) > 0:
    X_valid = np.array(X_v)
    y_valid = np.array(y_v)
X_train = np.array(X_train)
y_train = np.array(y_train)
write_y = (write_y[:, :2]).astype(int)
y_train = (y_train[:, :2]).astype(int)
y_valid = (y_valid[:, :2]).astype(int)
y_test = (y_test[:, :2]).astype(int)

print('Writing dataset')
with h5py.File("{}_dataset.h5".format(args['output']), 'w') as out:
    out.create_dataset('X_train', data=X_train)
    out.create_dataset('X_valid', data=X_valid)
    out.create_dataset('X_test', data=X_test)
    out.create_dataset('y_train', data=y_train)
    out.create_dataset('y_valid', data=y_valid)
    out.create_dataset('y_test', data=y_test)
    out.create_dataset('X', data=write_X)
    out.create_dataset('y', data=write_y) 
    out.create_dataset('sequence_id_map', data=json.dumps(sequence_id_map))

print('Done')

# __END__