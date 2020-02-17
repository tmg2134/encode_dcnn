# encode_dcnn
Convolutional net designed to discover enhancers and novel motifs
------------------------

I reccomend having virtualenv or conda installed an enviornment for this cnn

Required software:
- bedtools2
- Keras
- Tensorflow
- python (I used 3.7.3)
- sklearn
- numpy
- h5py
- pysam

Files to have ready:
- A bed file with the regions you want to train on ready. IE: my_peaks_from_HepG2.bed
- Appropriate reference genome fasta for your data
- List of known enhancer regions. Such as, a list of regions from ENCODE
- Appropriate Chromosome sizes file IE: hg19.chrom.canon.sorted2.sizes

## Run the code!!

###### Create a dataset:

```
  python encode_dcnn/model_creation/create_dataset.py \
  --peaks my_peaks_from_HepG2.bed \
  --output HepG2
  --bedtools-path software/bedtools2/bin/bedtools \
  --reference-fasta reference/gencode19/GRCh37.p13.genome.fa \
  --blacklist peaks/known_enhancers.bed \
  --genome-sizes references/hg19.chrom.canon.sorted2.sizes \
  --test-set 6 9 \
  --valid-set 2
```
This will create a file named HepG2_dataset.h5

###### Create a model

```
  python python encode_dcnn/model_creation/starr_seq_cnn.py \
  --data my_training_dataset.h5 \
  --output-stub HepG2 \
  --al 12 \
  --sl 500 \
  --alfs 5 \
  --if 15 \
  -e 25
```
Note: -e is epochs. This will automatically save the model with the best val_loss as well as model after the last epoch.
