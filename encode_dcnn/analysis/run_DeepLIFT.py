from __future__ import print_function
import deeplift
from keras.models import model_from_json
from deeplift.util import compile_func
from deeplift.layers import NonlinearMxtsMode
import deeplift.conversion.kerasapi_conversion as kc
from importlib import reload
reload(deeplift.layers)
reload(deeplift.conversion.kerasapi_conversion)
from collections import OrderedDict
import deeplift
from keras.models import model_from_json
import json
import h5py
import numpy as np
import argparse
import encode_dcnn.tools.save_not_show

"""
    Script to run DeepLIFT from: https://github.com/kundajelab/deeplift
"""

OHE_SEQ = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0]
}

def one_hot_encode_sequence(seq):
    return np.array([OHE_SEQ[s] for s in seq.upper()])

UN_OHE_SEQ = {
    0 : 'A',
    1 : 'C',
    2 : 'G',
    3 : 'T'
}

def un_one_hot_encode_sequence(OHE_SEQ_np):
    new_seqs = []
    n = 0
    for seq in OHE_SEQ_np:
        new_seq = ''
        for nuc in seq:
            if np.count_nonzero(nuc) != 0:
                nextNuc = UN_OHE_SEQ[list(nuc).index(1)]
            else:
                nextNuc = 'N'
                n += 1
            new_seq = new_seq + nextNuc
        new_seqs.append(new_seq)
    if n > 0:
        print(n)
    return new_seqs

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m',  help="model from cnn")
parser.add_argument(
    '--weights',
    '-w',
    help="weights file from running model",
    default='output_autoPrimerPicker'
)
parser.add_argument('--data', '-d', help="dataset", default=3)
parser.add_argument(
    '--out',
    default='deeplift_annotated motifs found',
    help='name of output files with annotated motifs found',
)
parser.add_argument('--numSeqs', default=1000)
parser.add_argument(
    '--motifs-bed',
    help='bed file of motifs to look for in this data'
)
args = vars(parser.parse_args())

#load the keras model
keras_model_weights = args['weights']
keras_model_json = args['model']
keras_model = model_from_json(open(keras_model_json).read())
keras_model.load_weights(keras_model_weights)
enhancer_data = h5py.File(args['data'])
output_img = args['out']
seqsToRun = args['numSeqs']
if output_img == '':
    output_img = args['data'].split("_")[0] + "."


method_to_model = OrderedDict()
for method_name, nonlinear_mxts_mode in [
    #The genomics default = rescale on conv layers, revealcance on fully-connected
    ('rescale_conv_revealcancel_fc', NonlinearMxtsMode.DeepLIFT_GenomicsDefault),
    ('rescale_all_layers', NonlinearMxtsMode.Rescale),
    ('revealcancel_all_layers', NonlinearMxtsMode.RevealCancel),
    ('grad_times_inp', NonlinearMxtsMode.Gradient),
    ('guided_backprop', NonlinearMxtsMode.GuidedBackprop)]:
    method_to_model[method_name] = kc.convert_model_from_saved_files(
        h5_file=keras_model_weights,
        json_file=keras_model_json,
        nonlinear_mxts_mode=nonlinear_mxts_mode)

X = np.array(enhancer_data['X'])
X_to_shuffle = un_one_hot_encode_sequence(X)
# embedding = json.loads(enhancer_data['sequence_id_map'])

# y = np.array(enhancer_data['y_test'])


# def sanity_check():
#    model_to_test = method_to_model['rescale_conv_revealcancel_fc']
#    deeplift_prediction_func = compile_func([model_to_test.get_layers()[0].get_activation_vars()],
#                                         model_to_test.get_layers()[-1].get_activation_vars())
#     original_model_predictions = keras_model.predict(X, batch_size=200)
#     converted_model_predictions = deeplift.util.run_function_in_batches(
#                                     input_data_list=[X],
#                                     func=deeplift_prediction_func,
#                                     batch_size=200,
#                                     progress_update=None)
#     print("maximum difference in predictions:",np.max(np.array(converted_model_predictions)-np.array(original_model_predictions)))
#     assert np.max(np.array(converted_model_predictions)-np.array(original_model_predictions)) < 10**-5
#     predictions = converted_model_predictions

# Importance scores
print("Compiling scoring functions")
method_to_scoring_func = OrderedDict()
for method,model in method_to_model.items():
    print("Compiling scoring function for: "+method)
    method_to_scoring_func[method] = model.get_target_contribs_func(find_scores_layer_idx=0,
                                                                    target_layer_idx=-2)
    
#To get a function that just gives the gradients, we use the multipliers of the Gradient model
gradient_func = method_to_model['grad_times_inp'].get_target_multipliers_func(find_scores_layer_idx=0,
                                                                              target_layer_idx=-2)
print("Compiling integrated givesradients scoring functions")
integrated_gradients10_func = deeplift.util.get_integrated_gradients_function(
    gradient_computation_function = gradient_func,
    num_intervals=10)
method_to_scoring_func['integrated_gradients10'] = integrated_gradients10_func

background = OrderedDict([('A', 0.3), ('C', 0.2), ('G', 0.2), ('T', 0.3)])

from collections import OrderedDict
method_to_task_to_scores = OrderedDict()
for method_name, score_func in method_to_scoring_func.items():
    print("on method",method_name)
    method_to_task_to_scores[method_name] = OrderedDict()
    for task_idx in [0,1]:
        scores = np.array(score_func(
                    task_idx=task_idx,
                    input_data_list=[X],
                    input_references_list=[
                     np.array([background['A'],
                               background['C'],
                               background['G'],
                               background['T']])[None,None,:]],
                    batch_size=128,
                    progress_update=None))
        assert scores.shape[2]==4
        #The sum over the ACGT axis in the code below is important! Recall that DeepLIFT
        # assigns contributions based on difference-from-reference; if
        # a position is [1,0,0,0] (i.e. 'A') in the actual sequence and [0.3, 0.2, 0.2, 0.3]
        # in the reference, importance will be assigned to the difference (1-0.3)
        # in the 'A' channel, (0-0.2) in the 'C' channel,
        # (0-0.2) in the G channel, and (0-0.3) in the T channel. You want to take the importance
        # on all four channels and sum them up, so that at visualization-time you can project the
        # total importance over all four channels onto the base that is actually present (i.e. the 'A'). If you
        # don't do this, your visualization will look very confusing as multiple bases will be highlighted at
        # every position and you won't know which base is the one that is actually present in the sequence!
        scores = np.sum(scores, axis=2)
        method_to_task_to_scores[method_name][task_idx] = scores

from deeplift.util import get_shuffle_seq_ref_function
#from deeplift.util import randomly_shuffle_seq
from deeplift.dinuc_shuffle import dinuc_shuffle #function to do a dinucleotide shuffle

rescale_conv_revealcancel_fc_many_refs_func = get_shuffle_seq_ref_function(
    #score_computation_function is the original function to compute scores
    score_computation_function=method_to_scoring_func['rescale_conv_revealcancel_fc'],
    #shuffle_func is the function that shuffles the sequence
    #technically, given the background of this simulation, randomly_shuffle_seq
    #makes more sense. However, on real data, a dinuc shuffle is advisable due to
    #the strong bias against CG dinucleotides
    shuffle_func=dinuc_shuffle,
    one_hot_func=lambda x: np.array([one_hot_encode_sequence(seq) for seq in x]))   

### TODO
### Parallelize
num_refs_per_seq=10 #number of references to generate per sequence
method_to_task_to_scores['rescale_conv_revealcancel_fc_multiref_'+str(num_refs_per_seq)] = OrderedDict()
for task_idx in [0,1]:
    #The sum over the ACGT axis in the code below is important! Recall that DeepLIFT
    # assigns contributions based on difference-from-reference; if
    # a position is [1,0,0,0] (i.e. 'A') in the actual sequence and [0, 1, 0, 0]
    # in the reference, importance will be assigned to the difference (1-0)
    # in the 'A' channel, and (0-1) in the 'C' channel. You want to take the importance
    # on all channels and sum them up, so that at visualization-time you can project the
    # total importance over all four channels onto the base that is actually present (i.e. the 'A'). If you
    # don't do this, your visualization will look very confusing as multiple bases will be highlighted at
    # every position and you won't know which base is the one that is actually present in the sequence!
    method_to_task_to_scores['rescale_conv_revealcancel_fc_multiref_'+str(num_refs_per_seq)][task_idx] =\
        np.sum(rescale_conv_revealcancel_fc_many_refs_func(
            task_idx=task_idx,
            input_data_sequences=X_to_shuffle,
            num_refs_per_seq=num_refs_per_seq,
            batch_size=128,
            progress_update=1000,
        ),axis=2)

sequence_id_map = json.loads(enhancer_data['sequence_id_map'][()])
theYs = enhancer_data['y']
known_motifs_bed = open(args['motifs_bed'], 'r')
known_motifs = []

for i in known_motifs_bed:
    known_motifs.append(':'.join(i.strip().split("\t")[:3]))

# known_motif_pointer = 0

method_name = 'grad_times_inp'
task = 0
this_seq_len = 125
for idx in range(len(scores)):
    if idx < len(scores):
        if theYs[idx][0] == 0:
            continue
        

        chrom, start, stop = sequence_id_map[str(idx)].strip().split(".")[0].split(':')
        #stop = int(stop)
        bases_from_start = int(sequence_id_map[str(idx)].strip().split(".")[1]) * 5
        start = int(start) + bases_from_start
        stop = start + this_seq_len
        highlight = {'blue':[]}
        for i in known_motifs:
            mChrom, mStart, mStop = i.split(":")
            mStart = int(mStart)
            mStop = int(mStop)
            if chrom == mChrom:
                mRegionLen = mStop - mStart
                if mStart >= start and mStop <= stop:                    
                    start_pos = mStart - start
                    end_pos = mStop - start
                    highlight['blue'].append([start_pos,end_pos])

        # only need to calculate scores and plot if a known_motif is in this region
        if len(highlight['blue']) > 0:
            scores = method_to_task_to_scores[method_name][task]
            scores_for_idx = scores[idx]
            original_onehot = X[idx]
            scores_for_idx = original_onehot*scores_for_idx[:,None]
            
            save_not_show.plot_weights(scores_for_idx, '{}{}_{}'.format(output_img,method_name,idx), subticks_frequency=10, highlight=highlight)
    else:
        break

# __END__