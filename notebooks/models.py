import tensorflow as tf
from tensorflow import keras

##
## high complexity, dual channel
##

def gen_dual_high(K, C, loss='mean_squared_error', optimizer='adam', pooling='AveragePooling2D'):
    Pooling2D = getattr(keras.layers, pooling)
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
    output = keras.layers.Dense(units=1)(layer8)
    model = keras.Model(inputs=[input_fid, input_map], outputs=[output])
    model.compile(loss=loss, optimizer=optimizer)
    return model

##
## medium complexity, dual channel
##

def gen_dual_medium(K, C, loss='mean_squared_error', optimizer='adam', pooling='AveragePooling2D'):
    Pooling2D = getattr(keras.layers, pooling)
    input_fid = keras.layers.Input(shape=(1,), name='fid')
    input_map = keras.layers.Input(shape=(K, K, C), name='map')
    layer1 = keras.layers.Conv2D(filters=8, kernel_size=8, activation='relu')(input_map)
    layer2 = Pooling2D(pool_size=8)(layer1)
    layer3 = keras.layers.Conv2D(filters=16, kernel_size=8, activation='relu')(layer2)
    layer4 = Pooling2D(pool_size=4)(layer3)
    layer5 = keras.layers.Flatten()(layer4)
    layer6 = keras.layers.Dropout(0.5)(layer5)
    layer7 = keras.layers.Dense(units=32, activation='relu')(layer6)
    layer8 = keras.layers.Dropout(0.5)(layer7)
    output = keras.layers.Dense(units=1)(layer8)
    model = keras.Model(inputs=[input_fid, input_map], outputs=[output])
    model.compile(loss=loss, optimizer=optimizer)
    return model

##
## low complexity, dual channel
##

def gen_dual_low(K, C, loss='mean_squared_error', optimizer='adam', pooling='AveragePooling2D'):
    Pooling2D = getattr(keras.layers, pooling)
    input_fid = keras.layers.Input(shape=(1,), name='fid')
    input_map = keras.layers.Input(shape=(K, K, C), name='map')
    layer1 = keras.layers.Conv2D(filters=4, kernel_size=8, activation='relu')(input_map)
    layer2 = Pooling2D(pool_size=8)(layer1)
    layer3 = keras.layers.Conv2D(filters=8, kernel_size=8, activation='relu')(layer2)
    layer4 = Pooling2D(pool_size=4)(layer3)
    layer5 = keras.layers.Flatten()(layer4)
    layer6 = keras.layers.Dropout(0.5)(layer5)
    layer7 = keras.layers.Dense(units=16, activation='relu')(layer6)
    layer8 = keras.layers.Dropout(0.5)(layer7)
    output = keras.layers.Dense(units=1)(layer8)
    model = keras.Model(inputs=[input_fid, input_map], outputs=[output])
    model.compile(loss=loss, optimizer=optimizer)
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
    output = keras.layers.Dense(units=1)(layer8)
    model = keras.Model(inputs=[input_fid, input_map], outputs=[output])
    model.compile(loss=loss, optimizer=optimizer)
    return model

##
## depthwise convolve, medium complexity
##

def gen_depth_medium(K, C, loss='mean_squared_error', optimizer='adam'):
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
    output = keras.layers.Dense(units=1)(layer8)
    model = keras.Model(inputs=[input_fid, input_map], outputs=[output])
    model.compile(loss=loss, optimizer=optimizer)
    return model

##
## depthwise convolve, low complexity
##

def gen_depth_low(K, C, loss='mean_squared_error', optimizer='adam'):
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
    output = keras.layers.Dense(units=1)(layer8)
    model = keras.Model(inputs=[input_fid, input_map], outputs=[output])
    model.compile(loss=loss, optimizer=optimizer)
    return model

##
## average density model
##

def gen_avg_density(K, C, loss='mean_squared_error', optimizer='adam'):
    input_fid = keras.layers.Input(shape=(1,), name='fid')
    input_map = keras.layers.Input(shape=(K, K, C), name='map')
    layer1 = keras.layers.GlobalAveragePooling2D()(input_map)
    output = keras.layers.Dense(units=1)(layer1)
    model = keras.Model(inputs=[input_fid, input_map], outputs=[output])
    model.compile(loss=loss, optimizer=optmizer)
    return model
    
##
## variable radius pooling model
##

class RadialPooling2D(keras.layers.Layer):
    def __init__(self, R, S, **kwargs):
        super().__init__(**kwargs)
        self.R = R
        self.S = S

    def build(self, input_shape):
        _, self.span_x, self.span_y, self.chan = input_shape
        self.size0 = self.add_weight(name='size', shape=(1,), dtype=tf.float32, initializer='uniform', trainable=True)
        super().build(input_shape)

    def call(self, x):
        size = self.R*keras.activations.sigmoid(self.size0)
        zero_x, zero_y = int(self.span_x//2), int(self.span_y//2)
        vals_x, vals_y = tf.cast(tf.range(self.span_x), tf.float32), tf.cast(tf.range(self.span_y), dtype=tf.float32)
        grid_x, grid_y = tf.meshgrid(vals_x, vals_y)
        radius = tf.sqrt((grid_x-zero_x)**2+(grid_y-zero_y)**2)
        mask = keras.activations.sigmoid(-(radius-size)/self.S)[None,:,:,None]
        return tf.reduce_mean(x*mask, axis=[1, 2])

    def compute_output_shape(self, input_shape):
        return (1,)

class RadialPooling2DX(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.size = self.add_weight(name='size', shape=(1,), initializer='uniform', trainable=True)
        super().build(input_shape)

    def call(self, x):
        size1 = keras.activations.sigmoid(self.size)
        return tf.reduce_mean(x*size1, axis=[1, 2])

    def compute_output_shape(self, input_shape):
        return (1,)

    
def gen_radial_pool(K, C, P, R=128, S=5, loss='mean_squared_error', optimizer='adam'):
    input_fid = keras.layers.Input(shape=(1,), name='fid')
    input_map = keras.layers.Input(shape=(K, K, C), name='map')
    pool = keras.layers.Concatenate()([RadialPooling2D(R, S)(input_map) for _ in range(P)])
    output = keras.layers.Dense(1)(pool)
    model = keras.Model(inputs=[input_fid, input_map], outputs=[output])
    model.compile(loss=loss, optimizer=optimizer)
    return model
