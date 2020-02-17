import os
import sys
import subprocess
from tempfile import TemporaryDirectory
import numpy as np

'''
Args:
  Takes a peaks file, blacklist, peak_length, and bedtools path as input

Return:
  a dict with list of complement regions and enhancer_regions (Use a negative set)

Description:
  Operations to clean up peaks and give out a final peak file and complement file

'''

def get_comp(peaks, blacklist, peak_length, genome_size, bedtools_path):
  tmpdir = TemporaryDirectory()
    EXPAND = peak_length

  # Expand peaks
  with open(os.path.join(tmpdir.name, 'expd_bed1.bed'), 'w') as expanded_bed1:
      for record in open(peaks):
          chrom, start, stop = record.strip().split('\t')[:3]
          midpoint = int(np.mean([int(start), int(stop)]))
          expanded_bed1.write('\t'.join([chrom, str(midpoint - EXPAND), 
            str(midpoint + EXPAND)]) + '\n')

  # Sort intersected regions
  # print('Sorting intersected regions')
  subprocess.call('{bedtools} sort -i {input} > {output}'
    .format(
      bedtools=bedtools_path,
      input=os.path.join(tmpdir.name, 'expd_bed1.bed'),
      output=os.path.join(tmpdir.name, 'expd_bed1_sorted.bed')
    ), 
    shell=True
  )

  # Merge intersected regions to remove any overlap
  # print('Merging overlaps')
  subprocess.call('{bedtools} merge -i {input} > {output}'
    .format(
      bedtools=bedtools_path,
      input=os.path.join(tmpdir.name, 'expd_bed1_sorted.bed'),
      output=os.path.join(tmpdir.name, 'expd_bed1_merged.bed')
    ), 
    shell=True
  )

  # Expand midpoint of merged intersected regions to 500bp
  # print('Expanding midpoints, phase 2')
  with open(os.path.join(tmpdir.name, 'bed_final.bed'), 'w') as bed_final:
      for record in open(os.path.join(tmpdir.name, 'expd_bed1_merged.bed')):
          chrom, start, stop = record.strip().split('\t')[:3]
          midpoint = int(np.mean([int(start), int(stop)]))
          bed_final.write('\t'.join([chrom, str(midpoint - EXPAND), 
            str(midpoint + EXPAND)]) + '\n')

  # print('Sorting intersected regions')
  subprocess.call('{bedtools} sort -i {input} > {output}'
    .format(
      bedtools=bedtools_path,
      input=os.path.join(tmpdir.name, 'bed_final.bed'),
      output=os.path.join(tmpdir.name, 'bed_final.sorted.bed')
    ), 
    shell=True
  )

  # Generate complement file
  # print('Generating complement file')
  subprocess.call(
    '{bedtools} complement -i {input} -g {genome_sizes} > {output}'
    .format(
      bedtools=bedtools_path,
      input=os.path.join(tmpdir.name, 'bed_final.sorted.bed'),
      genome_sizes=genome_size,
      output=os.path.join(tmpdir.name, 'bed_complement.bed')
    ), 
    shell=True
  )

  # Subtract blacklist regions from complement file
  subprocess.call('{bedtools} subtract -a {comp} -b {peaksU} > {output}'
    .format(
      bedtools=bedtools_path,
      comp=os.path.join(tmpdir.name, 'bed_complement.bed'),
      peaksU=blacklist,
      # output=os.path.join(tmpdir.name, 'bed_final_complement.bed')
      output=os.path.join(tmpdir.name, 'bed_final_complement.bed')
    ), shell=True)

  region_set = {}

  enhancer_peaks = []
  for i in open(os.path.join(tmpdir.name, 'bed_final.sorted.bed')):
    enhancer_peaks.append(i)


  complement_regions = []
  for i in open(os.path.join(tmpdir.name, 'bed_final_complement.bed')):
      complement_regions.append(i.strip().split('\t'))

  region_set['enhancer_peaks'] = enhancer_peaks
  region_set['complement_regions'] = complement_regions

  tmpdir.cleanup()

  return region_set