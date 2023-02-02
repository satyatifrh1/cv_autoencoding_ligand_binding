
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyemma
import pyemma.plots as mplt

import h5py
from keras.callbacks import ModelCheckpoint

npy_file = 'A_3.npy'  # Here our input data is: A_3.npy and having the ligand-com to all C-alpha Distances.
data_npy = np.load(npy_file)


####################################################################################################################

#Splitting of Training and Testing data:

data_npy_mod = np.array(data_npy)*1.0
#data_mod = np.concatenate(data_npy)*1.0

data_new1 = []
data_new2 = []
for i in range(0,3,1):  # If we proceed with 3 trajectories, then this will be 3 in the for loop variable.
    T1 = []
    T2 = []
    for j in range( np.shape(data_npy_mod[i])[0] ):
        if(j%10 == 0):                                # Since j%10 is done, so later on validation_split=0.1 will be used.
            T1.append( (data_npy_mod[i]) [j] )
        else:
            T2.append( (data_npy_mod[i]) [j] )
    TESTING = np.array(T1)*1.0
    TRAINING = np.array(T2)*1.0
    #print(i)
    data_new1.append(TRAINING)
    data_new2.append(TESTING)

print(np.shape(data_new1))
print(np.shape(np.concatenate(data_new1)))
print(np.shape(data_new2))
print(np.shape(np.concatenate(data_new2)))

print(np.shape(data_npy_mod))

data_new = []
data_new.append(np.concatenate(data_new1))
data_new.append(np.concatenate(data_new2))
data_new_final = np.concatenate(data_new)


####################################################################################################################

#Setting All the seeds necessity to get reproducible results while running in CPU:

import keras
from keras import layers
from keras.layers import Input,Dense,Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics

import tensorflow as tf

seed=12345              # random seed.
import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)



####################################################################################################################

#Sometimes the same block should be repeated twice to make sure the effect of these commands:
#So, the previous block is being reated again one more time.

import keras
from keras import layers
from keras.layers import Input,Dense,Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics

import tensorflow as tf

seed=12345
import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


####################################################################################################################

#This is for "Scaling" the data:
input_data = tf.keras.utils.normalize(data_new_final,axis=1)


#For T4L-Benzene, the ligand to C-alpha distances were 164.
#All these 164 dimensions would slowly compressed in a 2-D embeddings in the bottle-neck layer.

input_img = Input(shape=(164,))
encoded1 = Dense(72, activation='tanh')(input_img)
encoded2 = Dense(36, activation='linear')(encoded1)
encoded3 = Dense(12, activation='tanh')(encoded2)
encoded4 = Dense(2, activation='linear')(encoded3)
decoded3 = Dense(12, activation='tanh')(encoded4)
decoded2 = Dense(36, activation='linear')(decoded3)
decoded1 = Dense(72, activation='tanh')(decoded2)
decoded = Dense(164, activation='linear')(decoded1)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# checkpoint
# This is needed for checkpoint file generation.
checkpoint = ModelCheckpoint("model_{epoch:02d}.h5", verbose=1,mode='auto',period=5)
callbacks_list = [checkpoint]


#autoencoder.fit(input_data, input_data, epochs=2000,callbacks=callbacks_list,batch_size=1000)
AE = autoencoder.fit(input_data, input_data, epochs=30,callbacks=callbacks_list,batch_size=250, validation_split=0.1)

####################################################################################################################


#To get the output data:
output_data = autoencoder.predict(input_data)

#endata is the "encoded data" in the bottle-neck layer:
#We will need this endata for the rest of the analysis:
encoder = Model(autoencoder.input, autoencoder.layers[4].output)  # main important line
endata = encoder.predict(input_data)


#We have to save this "endata" for all our future work.

####################################################################################################################





