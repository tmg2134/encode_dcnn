# Motif discovery
import json
import h5py
import numpy as np
import sys
import argparse
import modisco
import modisco.backend
import modisco.nearest_neighbors
import modisco.affinitymat
import modisco.tfmodisco_workflow.seqlets_to_patterns
import modisco.tfmodisco_workflow.workflow
import modisco.aggregator
import modisco.cluster
import modisco.value_provider
import modisco.core
import modisco.coordproducers
import modisco.metaclusterers
import encode_dcnn.tools.save_not_show_viz

from collections import OrderedDict

"""
  Script to run TF-MoDISco from: https://github.com/kundajelab/tfmodisco
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

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', help="dataset used as input to model")
parser.add_argument('--deepLift-scores' ,'-ds')
# parser.add_argument('--fasta', '-f', help="Input sequences used to create dataset in fasta format")
parser.add_argument('--out', '-o', help="prefix for output names")
# parser.add_argument('--h5-results', '-r', help="Input if you just need graphs")
args = vars(parser.parse_args())

enhancer_scores = h5py.File(args['data'], 'r')
# fasta_seqs = open(args['fasta'],'r')

deepLift_scores = h5py.File(args['deepLift_scores'], 'r')

tasks = deepLift_scores["contrib_scores"].keys()

task_to_scores = OrderedDict()
task_to_hyp_scores = OrderedDict()

for task in tasks:
    #Note that the sequences can be of variable lengths;
    #in this example they all have the same length (200bp) but that is
    #not necessary.
    task_to_scores[task] = [np.array(x) for x in deepLift_scores['contrib_scores'][task]]
    task_to_hyp_scores[task] = [np.array(x) for x in deepLift_scores['hyp_contrib_scores'][task]]

X = enhancer_scores['X']
pos_seqs = []
theYs = enhancer_scores['y']
for idx in range(len(X)):
    if theYs[idx][0] == 1:
        pos_seqs.append(X[idx])

# onehot_data = [one_hot_encode_sequence(seq) for seq in fasta_seqs]

null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)

tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
  #Slight modifications from the default settings
  sliding_window_size=15,
  flank_size=5,
  target_seqlet_fdr=0.15,
  seqlets_to_patterns_factory=
   modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
      trim_to_window_size=15,
      initial_flank_to_add=5,
      kmer_len=5, num_gaps=1,
      num_mismatches=0,
      final_min_cluster_size=60)
  )(
  # task_names=["task0", "task1", "task2"],
  task_names=["task0"],
  contrib_scores=task_to_scores,
  hypothetical_contribs=task_to_hyp_scores,
  one_hot=pos_seqs,
  null_per_pos_scores = null_per_pos_scores)

grp = h5py.File("{}_results.hdf5".format(args['out']))
tfmodisco_results.save_hdf5(grp)
grp.close()

# plot stuff. # Can run seperately
from collections import Counter
from matplotlib import pyplot as plt

import modisco.affinitymat.core
import modisco.cluster.phenograph.core
import modisco.cluster.phenograph.cluster
import modisco.cluster.core

hdf5_results = h5py.File("{}_results.hdf5".format(args['out']),"r")

print("Metaclusters heatmap")
import seaborn as sns
activity_patterns = np.array(hdf5_results['metaclustering_results']['attribute_vectors'])[
                    np.array(
        [x[0] for x in sorted(
                enumerate(hdf5_results['metaclustering_results']['metacluster_indices']),
               key=lambda x: x[1])])]
sns.heatmap(activity_patterns, center=0)
plt.savefig("{}_heatmap.png".format(args['out']))

# plt.show()

metacluster_names = [
    x.decode("utf-8") for x in 
    list(hdf5_results["metaclustering_results"]
         ["all_metacluster_names"][:])]

all_patterns = []
background = np.array([0.27, 0.23, 0.23, 0.27])

for metacluster_name in metacluster_names:
    print(metacluster_name)
    metacluster_grp = (hdf5_results["metacluster_idx_to_submetacluster_results"]
                                   [metacluster_name])
    print("activity pattern:",metacluster_grp["activity_pattern"][:])
    all_pattern_names = [x.decode("utf-8") for x in 
                         list(metacluster_grp["seqlets_to_patterns_result"]
                                             ["patterns"]["all_pattern_names"][:])]
    if (len(all_pattern_names)==0):
        print("No motifs found for this activity pattern")
    for pattern_name in all_pattern_names:
        print(metacluster_name, pattern_name)
        all_patterns.append((metacluster_name, pattern_name))
        pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
        print("total seqlets:",len(pattern["seqlets_and_alnmts"]["seqlets"]))
        print("Task 0 hypothetical scores:")
        save_not_show_viz.plot_weights(pattern["task0_hypothetical_contribs"]["fwd"], plot_name='{}_task0_hyp_{}'.format(args['out'],pattern_name))
        print("Task 0 actual importance scores:")
        save_not_show_viz.plot_weights(pattern["task0_contrib_scores"]["fwd"], plot_name='{}_task0_contrib_{}'.format(args['out'],pattern_name))
        print("onehot, fwd and rev:")
        # save_not_show_viz.plot_weights(save_not_show_viz.ic_scale(
        #   np.array(pattern["sequence"]["fwd"]),  pattern_name),
        #   plot_name='{}_onehot_fwd_{}'.format(args['out'], background=background)
        # ) 
        # save_not_show_viz.plot_weights(save_not_show_viz.ic_scale(
        #   np.array(pattern["sequence"]["rev"]), pattern_name),
        #   plot_name='{}_onehot_rev_{}'.format(args['out'], background=background)
        # )
        
hdf5_results.close()

# Next Step?
from modisco.tfmodisco_workflow import workflow

track_set = modisco.tfmodisco_workflow.workflow.prep_track_set(
                task_names=tasks,
                contrib_scores=task_to_scores,
                hypothetical_contribs=task_to_hyp_scores,
                one_hot=pos_seqs)

grp = h5py.File("{}_results.hdf5".format(args['out']),"r")
loaded_tfmodisco_results =\
    workflow.TfModiscoResults.from_hdf5(grp, track_set=track_set)
grp.close()

# untrimmed_gata_pattern = (
#     loaded_tfmodisco_results
#     .metacluster_idx_to_submetacluster_results["metacluster_1"]
#     .seqlets_to_patterns_result.patterns[0])
# print("Untrimmed Gata - sequence (scaled by information content)")
# save_not_show_viz.plot_weights(save_not_show_viz.ic_scale(untrimmed_gata_pattern["sequence"].fwd, plot_name='UntrimmedGata', background=background))
# print("Untrimmed Gata - task 0 hypothetical scores")
# save_not_show_viz.plot_weights(untrimmed_gata_pattern["task0_hypothetical_contribs"].fwd)
# trimmed_gata = untrimmed_gata_pattern.trim_by_ic(ppm_track_name="sequence", plot_name='UntrimmedGatahyp'
#                                                  background=background,
#                                                  threshold=0.3)
# print("IC-trimmed Gata - sequence (scaled by information content)")
# save_not_show_viz.plot_weights(save_not_show_viz.ic_scale(trimmed_gata["sequence"].fwd, plot_name='IC_trimmedGata', background=background))

# __END__