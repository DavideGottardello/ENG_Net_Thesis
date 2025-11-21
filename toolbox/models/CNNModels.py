from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Flatten, GaussianNoise, Dropout
from keras.layers import Input, Dense, concatenate, LSTM, Masking, Conv2D, MaxPooling1D, Conv1D, ConvLSTM2D, Dropout
from keras.layers import MaxPooling2D, GlobalMaxPooling1D, Add, MaxPool2D, TimeDistributed, MaxPool3D, Input
from keras.models import Model
from keras.models import Sequential
import numpy as np
from keras.utils import plot_model
from tensorflow import keras

def CNN_4x(nb_classes = 4 , Chans = 16, Samples = 1000, 
             dropoutRate = 0.5, kernLength = 150):
    

    ##################################################################
    model = Sequential([
        
        Reshape((Chans // 4, Samples, 4), input_shape=(Chans, Samples)),
        
        Conv2D(filters=8, kernel_size=(Chans//4, kernLength), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(1, 10)),
        Dropout(dropoutRate//2.5),
        
        Conv2D(filters=16, kernel_size=(Chans//4, kernLength),  padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(Chans//4, 10)),
        Dropout(dropoutRate),
                      
        Flatten(name = 'flatten'),
        Dense(nb_classes, name = 'dense', kernel_constraint = max_norm(0.25)),
        Activation('softmax', name = 'softmax')
    
        
    ])
    return model

#################################################################################

def CNN_4xKL_v2(nb_classes = 4 , Chans = 16, Samples = 1000, 
             dropoutRate = 0.5, kernLength = 100):
    

    model = Sequential([
        
        Reshape((Chans // 4, Samples, 4), input_shape=(Chans, Samples)),
        
        Conv2D(filters=16, kernel_size=(Chans//4, kernLength), activation='relu', padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(1, 7)),
        Dropout(dropoutRate//2.5),
        
        Conv2D(filters=32, kernel_size=(Chans//4, kernLength//10), activation='relu', padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(1, 5)),
        Dropout(dropoutRate),
        
        Conv2D(filters=64, kernel_size=(Chans//4, kernLength//20), activation='relu', padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(4, 4)),
        Dropout(dropoutRate),
        
        Flatten(name = 'flatten'),
        Dense(nb_classes, name = 'dense', kernel_constraint = max_norm(0.25)),
        Activation('softmax', name = 'softmax')
    
        
    ])
    return model

#################################################################################







def TempConv(x, filters_out, k_temp):
    t = Conv2D(filters_out, kernel_size=(k_temp,1), strides=1, padding= 'same',activation="relu")(x)
    return t

# 1D convolution on the electrodes - convolution between all electrodes at the same time
def ElecConv(x, filters_out, elec_temp):
    e = Conv2D(filters_out, kernel_size=(1,elec_temp), strides=1, activation="relu")(x)
    return e

# CNN constructed to predict using the convolutions created
def CNN_temp_elec(sequence_size, num_classes):
    f_out = [32, 64, 128, 256]
    #f_out = [32, 64, 128]
    kernel_t= [9, 9, 9, 9]
    kernel_e = [16, 8, 4, 2]
    ker_t_final=3
    inputs = keras.Input(shape=sequence_size + (1,))
    
    #1st CNN - Temporal and electrodes convolution, add both together, then dropout layer and maxpooling
    t1 = TempConv(inputs, f_out[0], kernel_t[0])
    e1 = ElecConv(inputs, f_out[0], kernel_e[0])
    te1 = tf.keras.layers.Add()([t1, e1])
    te1=Dropout(0.1)(te1)
    out1 = keras.layers.MaxPool2D(pool_size=(2, 2))(te1)
    
    #2nd CNN - Temporal and electrodes convolution, add both together, then dropout layer and maxpooling
    t2 = TempConv(out1, f_out[1], kernel_t[1])
    e2 = ElecConv(out1, f_out[1], kernel_e[1])
    te2 = tf.keras.layers.Add()([t2, e2])
    te2=Dropout(0.1)(te2)
    out2 = keras.layers.MaxPool2D((2, 2))(te2)
    
    #3rd CNN - Temporal and electrodes convolution, add both together, then dropout layer and maxpooling
    t3 = TempConv(out2, f_out[2], kernel_t[2])
    e3 = ElecConv(out2, f_out[2], kernel_e[2])
    te3 = tf.keras.layers.Add()([t3, e3])
    te3=Dropout(0.1)(te3)
    out3 = keras.layers.MaxPool2D((2, 2))(te3)
    
    #4th CNN - Temporal and electrodes convolution, add both together, then dropout layer and maxpooling
    t4 = TempConv(out3, f_out[3], kernel_t[3])
    e4 = ElecConv(out3, f_out[3], kernel_e[3])
    te4 = tf.keras.layers.Add()([t4, e4])
    te4=Dropout(0.1)(te4)
    out4 = keras.layers.MaxPool2D((2, 2))(te4)
    
    # Classification part - 2D convolution with same padding, maxpooling and dropout
    cnn_fin1=Conv2D(f_out[1], kernel_size=(ker_t_final,1), strides=1, padding= 'same',activation="relu")(out4)
    max1_fin = keras.layers.MaxPool2D((15, 1),strides=7)(cnn_fin1)
    max1_fin=Dropout(0.1)(max1_fin)
    
    # Classification part - 2D convolution with same padding, maxpooling and dropout
    cnn_fin2=Conv2D(f_out[2], kernel_size=(ker_t_final,1), strides=1, padding= 'same',activation="relu")(max1_fin)
    #max2_fin = keras.layers.MaxPool2D((15, 1),strides=7)(cnn_fin2)
    if cnn_fin2.shape[1]>=15:
        max2_fin = keras.layers.MaxPool2D((15, 1),strides=7)(cnn_fin2)
    elif cnn_fin2.shape[1]>=6:
        max2_fin = keras.layers.MaxPool2D((cnn_fin2.shape[1], 1),strides=7)(cnn_fin2)
    else:
        max2_fin = keras.layers.MaxPool2D((cnn_fin2.shape[1], 1),strides=3)(cnn_fin2)
    max2_fin=Dropout(0.1)(max2_fin)
    
    # Classification part - Flatten and softmax layers in order to achieve an output   
    output_fin=Flatten()(max2_fin)
    
    outputs= Dense(4, activation='softmax')(output_fin)
    #outputs= Dense(2, activation='softmax')(output_fin)
    model = keras.models.Model(inputs, outputs, name="CNN_temp_elec")
    
    return model

#################################################################################à
def CONV_LSTM(sequence_size, num_classes):
    f_out = [16, 32, 64, 128]
    if sequence_size[1]==50:
        kernel_t=[11,11,5,1]
    else:
        kernel_t=[11,11,11,11]
    #kernel_t=[9,9,9,9]
    kernel_e = [3,3,3,3]
    #kernel_e=[16,8,4,2]
    
    inputs = keras.Input(shape=sequence_size + (1,)) #add the dim corresponding to "channels"
    
    #1st CONV-LSTM
    cnnlstm1=ConvLSTM2D(filters = f_out[0], kernel_size = (kernel_t[0],kernel_e[0]), activation = 'tanh',
                        data_format = "channels_last",recurrent_dropout=0.2, 
                        return_sequences=True)(inputs)
    out1 = MaxPool3D(pool_size=(1,2,1), data_format="channels_last")(cnnlstm1)
    
    #2nd CONV-LSTM
    cnnlstm2=ConvLSTM2D(filters = f_out[1], kernel_size = (kernel_t[1],kernel_e[1]), activation = 'tanh',
                        data_format = "channels_last",recurrent_dropout=0.2, 
                        return_sequences=True)(out1)
    #timed1=TimeDistributed(cnnlstm2)(out1)
    out2 = MaxPool3D(pool_size=(1,2,1), data_format="channels_last")(cnnlstm2)
    
    #3rd CONV-LSTM
    cnnlstm3=ConvLSTM2D(filters = f_out[2], kernel_size = (kernel_t[2],kernel_e[2]), activation = 'tanh',
                        data_format = "channels_last",recurrent_dropout=0.2, 
                        return_sequences=True)(out2)
    #timed2=TimeDistributed(cnnlstm3)(out2)
    out3 = MaxPool3D(pool_size=(1,2,1), data_format="channels_last")(cnnlstm3)
    
    output_temp=Flatten()(out3)
    outputs= Dense(4, activation='softmax')(output_temp)
    model = keras.models.Model(inputs, outputs, name="CONV_LSTM")
    
    return model

#################################################################################


class Classifier_INCEPTION:
    '''
        output_directory - directory where output results will be saved
        input_shape - the shape of each sample to be trained
        nb_classes - the total of classes that the model has to predict
        verbose - if the user wants the information during the training or not
        build - calls the method to build the model and saves it into model
        batch_size - size of the batch used for training.
        nb_filters - total of filters to be used
        use_residual - if the residual connections should be used or not
        use_bottleneck - if the bottleneck layer should be used or not
        depth - how many inception blocks will be used in sequence
        kernel_size - desired size for the kernel
        nb_epochs - total of epochs to train the model
    '''
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64,
                 nb_filters=16, use_residual=True, use_bottleneck=True, depth=6, kernel_size=14, nb_epochs=40):
       
        self.output_directory = output_directory
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 16
        self.nb_epochs = nb_epochs

        if build == True:
            # Calls the method to build the model of the InceptionTime
            self.model = self.build_model(input_shape, nb_classes)
            # Shows to the user the building blocks of the model
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            # Saves the initial weights of the model
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    ''' Base block of the InceptionTime - the Inception Module '''
    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        # Transform the input by using a bottleneck layer to reduce dimensionality
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # Determines the kernel sizes based on the biggest kernel size (in the case of 40, they are 10, 20 and 40)
        kernel_size_s = [self.kernel_size // (2**i) for i in range(3)]

        conv_list = []

        # Appends the result of each 1D convolution with the different kernel sizes
        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        # Creates a MaxPooling layer from the original vector in order to keep the original information
        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        # Includes the MaxPooling Layer to the convolution result
        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        # Concatenates the results
        x = keras.layers.Concatenate(axis=2)(conv_list)
        
        # Performs Batch Normalization
        x = keras.layers.BatchNormalization()(x)
        
        # Saves the activation of the layers
        x = keras.layers.Activation(activation='relu')(x)
        
        return x

    ''' Shortcut layer to execute the residual connection (since this is still a RNN) '''
    def _shortcut_layer(self, input_tensor, out_tensor):
        # Performs the 1D convolution from the input tensor
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        # Performs Batch Normalization
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
        # Adds this layer to the model
        x = keras.layers.Add()([shortcut_y, out_tensor])
        # Saves the activation of the layers
        x = keras.layers.Activation('relu')(x)
        
        return x

    ''' Builds the model and saves it in order to be trained '''
    def build_model(self, input_shape, nb_classes):
        
        # Input layer that receives an input with input_shape
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        # Depth indicates how many inception modules are being concatenated.
        # In the general case, they are 6 modules
        for d in range(self.depth):
            # Includes a new inception module in the model
            x = self._inception_module(x)

            # Includes a residual connection between the first convolution block and
            # the one being considered (every 3 blocks)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x
        
        # Includes a Global Average Pooling in order to perform the prediction
        gap_layer = keras.layers.GlobalAveragePooling1D()(x)
        
        # Dense - Transforms all that was generated in a softmax prediction
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        # Creates the model with input and output layers
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        # Creates the model with its characteristics
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        # Function to reduce the learning rate if necessary
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10,
                                                      min_lr=0.0001)
        
        # Function to stop training early if no improvement is made
        early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, start_from_epoch=8, verbose=1)


        # Creates the path to save the best model
        file_path = self.output_directory + 'best_model.hdf5'

        # Creates the checkpoint in order to save the best model according to the metrics to monitor
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_accuracy',
                                                           save_best_only=True)

        # Create the callbacks based on the learning rate control and saving the best model
        self.callbacks = [reduce_lr, model_checkpoint, early_stop]

        return model

    ''' Actually trains the model fitting it to the data provided '''
    def fit(self, x_train, y_train, x_val, y_val, y_true, plot_test_acc=True):
        #if len(keras.backend.tensorflow_backend._get_available_gpus()) == 0:
         #   print('error no gpu')
          #  exit()
        
        # x_val and y_val are only used to monitor the test loss and NOT for training

        # Defines the mini batch size based on whether the user has provided a value or not
        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0]/10,16))
        else:
            mini_batch_size = self.batch_size
        
        # Fits the model to the data with the desired epochs and considering the given validation data to save the model
        if plot_test_acc:
            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        else:
            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, callbacks=self.callbacks)
            
        return hist
        
    ''' Generates the prediction based on the model that was trained'''
    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        return True