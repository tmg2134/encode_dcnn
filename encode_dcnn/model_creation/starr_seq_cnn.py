import os
import sys
import argparse
import h5py
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Convolution1D, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import Adam, Adadelta

# Load input and output data
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='dataset.h5')
parser.add_argument('--subsample-rate', type=float, default=1.0)
parser.add_argument('--output-stub', '--out', default='encode_cnn')
parser.add_argument('--seq-len', '--sl', type=int, default=500)
parser.add_argument('--addLayers', '--al', type=int, default=12)
parser.add_argument('--iFilter-size', '--if', type=int, default=15)
parser.add_argument('--addLayers-filter-size', '--alfs', type=int, default=5)
parser.add_argument('--loss', default='categorical_crossentropy')
parser.add_argument('--activation-func', '-a', default='relu')
parser.add_argument('--kernel-init', '-ki', default='he_uniform')
parser.add_argument('--learning-rate', '-lr', type=float, default=0.00025)
parser.add_argument('--batch-size', '-bs', type=int, default=32)
parser.add_argument('--epochs', '-e', type=int, default=30)
args = vars(parser.parse_args())

sequence_input_length = args['seq_len']
initial_filter = args['iFilter_size']
additional_layers = args['addLayers']
filter_size = args['addLayers_filter_size']
loss_function = args['loss']
kernel_init = args['kernel_init']
activation_func = args['activation_func']

default_layers = 2

snppet_net = Sequential()
# Initial Convolution Layer
snppet_net.add(Convolution1D(
    name='convolution1d_10',
    batch_input_shape=[None, sequence_input_length, 4],
    filters=120,  # nb_filter
    kernel_size=initial_filter,  # filter_length
    activation=activation_func,
    kernel_initializer=kernel_init))
snppet_net.add(BatchNormalization(name='batchnormalization_10', epsilon=1e-3))
snppet_net.add(Dropout(name='dropout_10', rate=0.1))

# Additional convolutions
for i in range(default_layers+additional_layers):
  snppet_net.add(Convolution1D(
    name='convolution1d_{}'.format(i+11),
    filters=120,
    kernel_size=filter_size,
    activation=activation_func,
    kernel_initializer=kernel_init))
  snppet_net.add(BatchNormalization(name='batchnormalization_{}'.format(i+11), epsilon=1e-3))
  snppet_net.add(Dropout(name='dropout_{}'.format(i+11), rate=0.1))

# Final
snppet_net.add(Flatten(name='flatten_4'))
snppet_net.add(Dense(name='dense_4', units=2, activation='softmax'))

# Compile with ADAM optimizer, mean squared error loss function
snppet_net.compile(loss=loss_function, optimizer=Adam(lr=.00015), metrics=['accuracy'])
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, 
#                                             factor=0.5, min_lr=0.00001)

enhancer_data = h5py.File(args['data'])
X_train = np.array(enhancer_data['X_train'])
y_train = np.array(enhancer_data['y_train'])
X_valid = np.array(enhancer_data['X_valid'])
y_valid = np.array(enhancer_data['y_valid'])
X_test = np.array(enhancer_data['X_test'])
y_test = np.array(enhancer_data['y_test'])

filepath = args['output_stub'] + '_fullmodel_best_val_loss.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Run the model
snppet_net.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=args['epochs'], 
               batch_size=args['batch_size'], callbacks=callbacks_list)
metrics = snppet_net.evaluate(X_test, y_test, batch_size=128)
print('Test metrics: {}'.format(metrics)) 

# Save the model weights
snppet_net.save(args['output_stub'] + '_fullmodel_final.h5')
with open(args['output_stub'] + '_model.json', 'w') as jsonout:
  jsonout.write(snppet_net.to_json())
snppet_net.save_weights(args['output_stub'] + '_modelweights.h5')

# __END__