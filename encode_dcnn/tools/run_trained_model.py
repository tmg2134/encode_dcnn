import argparse, h5py, sys, os, json
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Convolution1D, Flatten
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
# load and evaluate a saved model
from numpy import loadtxt
from sklearn.model_selection import cross_validate
from sklearn import datasets, svm
from sklearn.metrics import confusion_matrix,precision_recall_curve, matthews_corrcoef,auc,roc_curve, classification_report 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--model', 
  default='/cephfs/users/tgoodman/snnpet_cnn/encode_cnn_fullmodel.h5')
parser.add_argument('--weights', 
  default='/cephfs/users/tgoodman/snppet_cnn/encode_cnn_modelweights.h5')
parser.add_argument('--data', 
  default='/cephfs/users/tgoodman/snppet_cnn/dataset.hdf5')
parser.add_argument('--tset', default=False, action='store_true', 
  help="Use entire set in evaluation")
parser.add_argument('--out', '-o', 
  help="name and output a file of regions predicted as enhancers\n"+
       "\tBut only if they were marked as positive"
)
args = vars(parser.parse_args())

trained_model = load_model(args['model'])
enhancer_data = h5py.File(args['data'])
tset = args['tset']
#sys.stdout.write("Data loaded\n")

# from https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# split into input (X) and output (Y) variables
if tset:
  X = np.array(enhancer_data['X'])
  y = np.array(enhancer_data['y'])
else:
  X = np.array(enhancer_data['X_test'])
  y = np.array(enhancer_data['y_test'])

# evaluate the model
score = trained_model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (trained_model.metrics_names[1], score[1]*100))
y_pred = trained_model.predict(X,batch_size=128)
mcc = matthews_corrcoef(y.argmax(axis=1), y_pred.argmax(axis=1))


tp,fn,fp,tn = confusion_matrix(y.argmax(axis=1),y_pred.argmax(axis=1)).ravel()

recall = tp/(tp+fn)
precision = tp/(tp+fp)

print("TN: {}\nFN: {}\nFP: {}\nTP: {}".format(tn,fn,fp,tp))
print("Recall: %.4f\nPrecision: %.4f\n" % (recall,precision))
print("MCC: %.4f" % (mcc))

# ---------------------------------------------------------------------
# Get regions that were predicted to be enhancers regions
# and write them to a tsv file
# ---------------------------------------------------------------------

if args['out']:
  sequence_id_map = json.loads(enhancer_data['sequence_id_map'][()])
  predicted_enhancers = []
  for i,j in enumerate(y_pred):
    if j[0] >= .5 and y[i][0] == 1:
      predicted_enhancers.append(
        sequence_id_map[str(i)].strip().split(".")[0].split(':')
      )

  predicted_enhancers_file = open(args['out'], 'w')
  for i in predicted_enhancers:
    predicted_enhancers_file.write('\t'.join(i) + '\n')

# __END__