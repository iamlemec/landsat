import tensorflow as tf
from tensorflow import keras

##
## high complexity, dual channel
##

def gen_dual_high(K, C):
    input_fid = keras.layers.Input(shape=(1,), name='fid')
    input_map = keras.layers.Input(shape=(K, K, C), name='map')
    layer1 = keras.layers.Conv2D(filters=16, kernel_size=8, activation='relu')(input_map)
    layer2 = keras.layers.MaxPooling2D(pool_size=8)(layer1)
    layer3 = keras.layers.Conv2D(filters=32, kernel_size=8, activation='relu')(layer2)
    layer4 = keras.layers.MaxPooling2D(pool_size=4)(layer3)
    layer5 = keras.layers.Flatten()(layer4)
    layer6 = keras.layers.Dropout(0.5)(layer5)
    layer7 = keras.layers.Dense(units=64, activation='relu')(layer6)
    layer8 = keras.layers.Dropout(0.5)(layer7)
    output_prod = keras.layers.Dense(units=1)(layer8)
    model = keras.Model(inputs=[input_fid, input_map], outputs=[output_prod])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

##
## low complexity, dual channel
##

def gen_dual_low(K, C):
    input_fid = keras.layers.Input(shape=(1,), name='fid')
    input_map = keras.layers.Input(shape=(K, K, C), name='map')
    layer1 = keras.layers.Conv2D(filters=16, kernel_size=8, activation='relu')(input_map)
    layer2 = keras.layers.MaxPooling2D(pool_size=8)(layer1)
    layer3 = keras.layers.Conv2D(filters=16, kernel_size=8, activation='relu')(layer2)
    layer4 = keras.layers.MaxPooling2D(pool_size=4)(layer3)
    layer5 = keras.layers.Flatten()(layer4)
    layer6 = keras.layers.Dropout(0.5)(layer5)
    layer7 = keras.layers.Dense(units=32, activation='relu')(layer6)
    layer8 = keras.layers.Dropout(0.5)(layer7)
    output_prod = keras.layers.Dense(units=1)(layer8)
    model = keras.Model(inputs=[input_fid, input_map], outputs=[output_prod])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

##
## depthwise convolve, high complexity
##

def gen_depth_high(K, C, loss='mean_squared_error', optimizer='adam'):
    input_fid = keras.layers.Input(shape=(1,), name='fid')
    input_map = keras.layers.Input(shape=(K, K, C), name='map')
    layer1 = keras.layers.DepthwiseConv2D(depth_multiplier=8, kernel_size=8, activation='relu')(input_map)
    layer2 = keras.layers.MaxPooling2D(pool_size=8)(layer1)
    layer3 = keras.layers.DepthwiseConv2D(depth_multiplier=4, kernel_size=4, activation='relu')(layer2)
    layer4 = keras.layers.MaxPooling2D(pool_size=4)(layer3)
    layer5 = keras.layers.Flatten()(layer4)
    layer6 = keras.layers.Dropout(0.5)(layer5)
    layer7 = keras.layers.Dense(units=64, activation='relu')(layer6)
    layer8 = keras.layers.Dropout(0.5)(layer7)
    output_prod = keras.layers.Dense(units=1)(layer8)
    model = keras.Model(inputs=[input_fid, input_map], outputs=[output_prod])
    model.compile(loss=loss, optimizer=optimizer)
    return model

##
## depthwise convolve, medium complexity
##

def gen_depth_medium(K, C):
    input_fid = keras.layers.Input(shape=(1,), name='fid')
    input_map = keras.layers.Input(shape=(K, K, C), name='map')
    layer1 = keras.layers.DepthwiseConv2D(depth_multiplier=4, kernel_size=8, activation='relu')(input_map)
    layer2 = keras.layers.MaxPooling2D(pool_size=8)(layer1)
    layer3 = keras.layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=4, activation='relu')(layer2)
    layer4 = keras.layers.MaxPooling2D(pool_size=4)(layer3)
    layer5 = keras.layers.Flatten()(layer4)
    layer6 = keras.layers.Dropout(0.5)(layer5)
    layer7 = keras.layers.Dense(units=32, activation='relu')(layer6)
    layer8 = keras.layers.Dropout(0.5)(layer7)
    output_prod = keras.layers.Dense(units=1)(layer8)
    model = keras.Model(inputs=[input_fid, input_map], outputs=[output_prod])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

##
## depthwise convolve, low complexity
##

def gen_depth_low(K, C):
    input_fid = keras.layers.Input(shape=(1,), name='fid')
    input_map = keras.layers.Input(shape=(K, K, C), name='map')
    layer1 = keras.layers.DepthwiseConv2D(depth_multiplier=4, kernel_size=4, activation='relu')(input_map)
    layer2 = keras.layers.MaxPooling2D(pool_size=8)(layer1)
    layer3 = keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=4, activation='relu')(layer2)
    layer4 = keras.layers.MaxPooling2D(pool_size=7)(layer3)
    layer5 = keras.layers.Flatten()(layer4)
    layer6 = keras.layers.Dropout(0.5)(layer5)
    layer7 = keras.layers.Dense(units=16, activation='relu')(layer6)
    layer8 = keras.layers.Dropout(0.5)(layer7)
    output_prod = keras.layers.Dense(units=1)(layer8)
    model = keras.Model(inputs=[input_fid, input_map], outputs=[output_prod])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


##
## average density model (1024px)
##

def gen_mono_avg(K):
    input_fid = keras.layers.Input(shape=(1,), name='firm_id')
    input_dense = keras.layers.Input(shape=(K, K, 1), name='density_1024')
    output_prod = keras.layers.GlobalAveragePooling2D()(input_dense)
    model_simple = keras.Model(inputs=[input_fid, input_dense], outputs=[output_prod])
    model_simple.compile(loss='mean_squared_error', optimizer='adam')
