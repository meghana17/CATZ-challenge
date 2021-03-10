'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

#from prednet import PredNet
#from kitti_settings import *


# KITTI data saved here if you run process_kitti.py
DATA_DIR = '/content/catz_hkl_AUG'


# Model weights and config saved here if you run kitti_train.py
WEIGHTS_DIR = './model_data_keras2/'

# Results (prediction plots and evaluation file) saved here
RESULTS_SAVE_DIR = './catz_results/'

save_model = True  
#Contain only 5 images
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights_train4.hdf5')  
weights_file2 = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights_train4.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')



# Data files
train_file = os.path.join('/content/train_SPECIAL2_X_train.hkl')
train_sources = os.path.join('/content/train_SPECIAL2_sources_train.hkl')

val_file = os.path.join('/content/train_SPECIAL2_X_train.hkl')
val_sources = os.path.join('/content/train_SPECIAL2_sources_train.hkl')

print(hkl.load(train_sources))
print(len(list(set(hkl.load(train_sources)))))
print(len(list(set(hkl.load(val_sources)))))
# Training parameters
nb_epoch = 150
batch_size = 16

samples_per_epoch = (2433)*3
N_seq_val = 330*2 # number of sequences to use for validation
nt = 3  # number of timesteps used for sequences in training

# Model parameters
n_channels, im_height, im_width = (3, 96, 96)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
stack_sizes = (n_channels,48,96,192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
#CHANGED THE LOSS TO 0.5 IN THE LAST LAYER
layer_loss_weights = np.array([1., 0.1, 0.1, 0.1])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)

time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0


prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')

train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

lr_schedule = lambda epoch: 0.001 if epoch < 100 else 0.0001   # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
from keras.callbacks import Callback
class ImageCallback(Callback):
    not_print  ='False'
    def on_epoch_end(self, epoch, logs):
        print("EPOCH_CHECK",epoch,logs['loss'])
    
callbacks = [LearningRateScheduler(lr_schedule),ImageCallback()]


save_model ='True'
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='loss', save_best_only=True))
    #callbacks.append(ModelCheckpoint(filepath=weights_file2, monitor='val_loss', save_best_only=True))
    

history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,verbose=0,
validation_data=val_generator, validation_steps=N_seq_val / batch_size)

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
    
