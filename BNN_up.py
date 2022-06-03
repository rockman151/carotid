from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
import tensorflow_probability as tfp
from tensorflow.keras import layers
tf.keras.backend.set_image_data_format('channels_last')
tf.config.experimental.list_physical_devices('GPU')
import scipy.io as sio
########### cnn parameters
nb_ch=5
output_ch = 4
nb_fils=64
nb_mc=10
weight_decay=0.05
batch_size=2
_smooth = 1
img_size = 128
dr = 0.1 #dropout
filter_size = 3

def conv2d(num_filter, filter_size, stride_size, input_layer, bias_ct=0.01,activation=tf.keras.activations.relu):
    if activation is not None:
        layer = layers.Conv2D(num_filter, (filter_size, filter_size), # num. of filters and kernel size
                       strides=stride_size,
                       padding='same',
                       use_bias=True,
                       activation=None,
                       kernel_initializer='glorot_normal', # Xavier init
                       bias_initializer=tf.initializers.Constant(value=bias_ct))
        layer_o = layer(input_layer)
        layer_o = layers.BatchNormalization(axis=-1)(layer_o)
        #layer_o = layers.LeakyReLU(alpha=0.01)(layer_o)
        layer_o = tf.keras.activations.swish(layer_o)
        weights = layer.get_weights()
        bias = layer.bias
    else:
        layer = layers.Conv2D(num_filter, (filter_size, filter_size),  # num. of filters and kernel size
                              strides=stride_size,
                              padding='same',
                              use_bias=True,
                              activation=None,
                              kernel_initializer='glorot_normal',  # Xavier init
                              bias_initializer=tf.initializers.Constant(value=bias_ct))
        layer_o = layer(input_layer)
        layer_o = layers.BatchNormalization(axis=-1)(layer_o)
        layer_o = tf.keras.activations.linear(layer_o)
        weights = layer.get_weights()
        bias = layer.bias
    return layer_o,weights


def conv2d_trans(num_filter, filter_size, stride_size, input_layer, bias_ct=0.01,activation=tf.keras.activations.relu):
    if activation is not None:
        layer = layers.Conv2DTranspose(num_filter, (filter_size, filter_size),  # num. of filters and kernel size
                                strides=stride_size,
                                padding='same',
                                use_bias=True,
                                activation=None,
                                kernel_initializer='glorot_normal',  # Xavier init
                                bias_initializer=tf.initializers.Constant(value=bias_ct)
                                )

        layer_o = layer(input_layer)
        layer_o = layers.BatchNormalization(axis=-1)(layer_o)
        layer_o = tf.keras.activations.swish(layer_o)
        weights = layer.get_weights()
        bias = layer.bias
        return layer_o, weights, bias
    else:
        layer = layers.Conv2DTranspose(num_filter, (filter_size, filter_size),  # num. of filters and kernel size
                                       strides=stride_size,
                                       padding='same',
                                       use_bias=True,
                                       activation=None,
                                       kernel_initializer='glorot_normal',  # Xavier init
                                       bias_initializer=tf.initializers.Constant(value=bias_ct)
                                       )

        layer_o = layer(input_layer)
        layer_o = layers.BatchNormalization(axis=-1)(layer_o)
        weights = layer.get_weights()
        bias = layer.bias
        return layer_o, weights, bias
def dense(units,input_layer,activation = 'relu'):
    if activation =='relu':
        layer = layers.Dense(
        units, activation=None, use_bias=True, kernel_initializer='glorot_normal',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
    )

        layer_o = layer(input_layer)
        layer_o = tf.keras.activations.swish(layer_o)
    elif activation == None:
        layer = layers.Dense(
            units, activation=None, use_bias=True, kernel_initializer='glorot_normal',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
        )

        layer_o = layer(input_layer)
    elif activation =='linear':
        layer = layers.Dense(
            units, activation=None, use_bias=True, kernel_initializer='glorot_normal',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
        )

        layer_o = layer(input_layer)
        layer_o = tf.keras.activations.linear(layer_o)
    return layer_o
def dense_prob(units,input_layer):
    layer = tfp.layers.DenseFlipout(
    units, activation=None
)

    layer_o = layer(input_layer)
    return layer_o

def conv2d_prob(num_filter, filter_size, stride_size, input_layer,activation='relu'):
    layer = tfp.layers.Convolution2DFlipout(num_filter, (filter_size, filter_size), # num. of filters and kernel size
                   strides=stride_size,
                   padding='same',
                   activation=None,

                  )
    layer_o = layer(input_layer)
    layer_o = layers.BatchNormalization(axis=-1)(layer_o)
    if activation != None:
        layer_o = tf.keras.activations.swish(layer_o)
    weights = layer.get_weights()
    return layer_o,weights

def dense_net(input):
    fc_input = tf.reshape(input, (batch_size, -1, input.shape[3]))
    fc0 = dense(2, fc_input)
    fc1 = dense(2, fc0)
    fc2 = dense(2, fc1)
    fc3 = dense(2, fc2)
    fc4 = dense(output_ch, fc3,activation='linear')
    fc5 = tf.reshape(fc4, (batch_size, img_size, img_size, output_ch))
    output = fc5
    return output
def max_pooling(input):

    maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    out = maxpool(input)
    return out
input_x = tf.keras.Input(shape=(img_size,img_size,nb_ch),batch_size = batch_size)
c1,w1 = conv2d(nb_fils,filter_size,(1,1),input_x)
c1p,w1p = conv2d(nb_fils,filter_size,(1,1),c1)
c2,w2 = conv2d(nb_fils,filter_size,(2,2),c1p)
c2p,w2p = conv2d(nb_fils*2,filter_size,(1,1),c2)
c2d,w2d = conv2d(nb_fils*2,filter_size,(1,1),c2p)
c3,w3 = conv2d(nb_fils*2,filter_size,(2,2),c2d)
c3p, w3p = conv2d(nb_fils*4,filter_size,(1,1),c3)#drop
c3d, w3d = conv2d(nb_fils*4,filter_size,(1,1),c3p)#drop
c4, w4 = conv2d(nb_fils*4,filter_size,(2,2),c3d)
c4p, w4p = conv2d(nb_fils*8,filter_size,(1,1),c4)#drop
c4d, w4d = conv2d(nb_fils*8,filter_size,(1,1),c4p)#drop
c5, w5 = conv2d(nb_fils*8,filter_size,(2,2),c4d)
c5p, w5p = conv2d(nb_fils*16,filter_size,(1,1),c5)
c5d, w5d = conv2d(nb_fils*16,filter_size,(1,1),c5p)
d4, dw4, db4 =conv2d_trans(nb_fils*8,filter_size,(2,2),c5d)
d4c, dw4c = conv2d(nb_fils*8,filter_size,(1,1),tf.concat([c4d, d4],3))#drop
d4d, dw4d = conv2d_prob(nb_fils*8,filter_size,(1,1),d4c)#prob
d3, dw3, db3 = conv2d_trans(nb_fils*4,filter_size,(2,2),d4d)
d3c, dw3c = conv2d(nb_fils*4,filter_size,(1,1),tf.concat([c3d, d3],3))#drop
d3d, dw3d = conv2d_prob(nb_fils*4,filter_size,(1,1),d3c)#prob
d2, dw2, db2 = conv2d_trans(nb_fils*2,filter_size,(2,2),d3d)
d2c, dw2c = conv2d(nb_fils,filter_size,(1,1),tf.concat([c2d, d2],3))
d2d, dw2d = conv2d_prob(nb_fils,filter_size,(1,1),d2c)#prob
d1, dw1, db1 = conv2d_trans(nb_fils,filter_size,(2,2),d2d)#drop\
d1c, dw1c= conv2d(nb_fils,filter_size,(1,1),tf.concat([c1p, d1],3))
d1d, dw1d= conv2d_prob(nb_fils,filter_size,(1,1),d1c)#prob
d0c, dw0c= conv2d(nb_fils,filter_size,(1,1),d1d)
d0d, dw0d= conv2d(output_ch,1,(1,1),d0c,activation=None)
output = d0d
def dice_coef1(y_true, y_pred):

    y_pred = tf.nn.softmax(y_pred)
    y_pred_1 = y_pred[:,:,:,1:2]
    fac = tf.cast(y_pred_1 > 0.008, dtype=tf.int32)
    fac = tf.cast(fac, dtype=tf.float32)
    y_pred_1 = tf.multiply(y_pred_1, fac)
    y_pred_1 = tf.cast(y_pred_1, dtype=tf.float32)
    y_pred_2 = 1-y_pred_1
    y_true_1 = y_true[:,:,:,1:2]
    tmp_TP = tf.minimum(y_pred_1, y_true_1)
    TP = tf.reduce_sum(tmp_TP, [1, 2])
    tmp_FP = tf.maximum(y_pred_1 - y_true_1, 0)
    FP = tf.reduce_sum(tmp_FP, [1, 2])
    y_true2 = 1-y_true_1
    tmp_FN = tf.maximum(y_pred_2 - y_true2, 0)
    FN = tf.reduce_sum(tmp_FN, [1, 2])

    nominator = tf.multiply(TP, 2)
    tmp_denominator = tf.add(FP, FN)
    denominator = tf.add(tmp_denominator, tf.multiply(TP, 2))+1e-10
    fuzzy_dice = tf.reduce_mean(tf.divide(nominator, denominator))
    return fuzzy_dice

def loss_fn(y_true, y_pred):
    smooth = 1.
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_true = tf.stop_gradient(y_true)
    mc_x = (tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred))
    y_pred = tf.nn.softmax(y_pred)
    intersection = tf.math.reduce_sum(tf.math.abs(y_true * y_pred), axis=-1)
    dice = (2. * intersection + smooth) / (
                tf.math.reduce_sum(tf.math.square(y_true), -1) + tf.math.reduce_sum(tf.math.square(y_pred),
                                                                                    -1) + smooth)
    dice_loss = (1 - dice)
    KLD = (tf.keras.losses.KLD(
        y_true, y_pred)
    )
    loss = 0.1*dice_loss +0.8*mc_x+0.1*KLD
    return loss
############################################
############################################
model = tf.keras.Model(inputs=input_x, outputs=output)
opt=tf.keras.optimizers.Adam(
    learning_rate=0.0005, beta_1=0.9, beta_2=0.9999, epsilon=1e-07, amsgrad=False,
    name='Adam'
)
model.compile(loss = loss_fn, optimizer = opt, metrics = [[dice_coef1]],experimental_run_tf_function=False)
X_all = 'C:\\Users\\ran.li\PycharmProjects\\carotid_seg\\to_liran\\label_img_aug.mat'
X_all = sio.loadmat(X_all)
X_all = np.array(X_all['img'])
X_test = 'C:\\Users\\ran.li\PycharmProjects\\carotid_seg\\to_liran\\label_test_img.mat'
X_test = sio.loadmat(X_test)
X_test = np.array(X_test['img_test'])
X_test2 = 'C:\\Users\\ran.li\PycharmProjects\\carotid_seg\\to_liran\\img_test_canf.mat'
X_test2 = sio.loadmat(X_test2)
X_test2 = np.array(X_test2['img_test'])
X_test2 = X_test2[0:484,:,:,:]
X_train = X_all[:,:,:,0:nb_ch]
y_all = 'C:\\Users\\ran.li\PycharmProjects\\carotid_seg\\to_liran\\label_seg_aug.mat'
y_all = sio.loadmat(y_all)
y_all = np.array(y_all['seg'])
y_test = 'C:\\Users\\ran.li\PycharmProjects\\carotid_seg\\to_liran\\label_test_seg.mat'
y_test = sio.loadmat(y_test)
y_test = np.array(y_test['seg_test'])
y_test = y_test[:,:,:,:]
y_train = y_all[:,:,:,:]
y_test2 = 'C:\\Users\\ran.li\PycharmProjects\\carotid_seg\\to_liran\\seg_test_canf_hemor.mat'
y_test2 = sio.loadmat(y_test2)
y_test2 = np.array(y_test2['seg_test'])
y_test2 = y_test2[0:484,:,:,:]
k_X_train, k_X_val, k_y_train, k_y_val = train_test_split(X_train, y_train, test_size=20/420)
log_dir = 'C:\\Users\\ran.li\PycharmProjects\\carotid_seg\\to_liran\\tensorboard'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
for i in range(10):
    if i>0:
        model.load_weights('C:\\Users\\ran.li\PycharmProjects\\carotid_seg\\to_liran\\checkpoint_carotid\\train_bayesian_best_carotid2')
    history = model.fit(k_X_train,k_y_train,batch_size=batch_size, epochs=10,validation_data=(k_X_val, k_y_val), verbose=1, shuffle=True,callbacks=[tensorboard_callback])
    model.save_weights('C:\\Users\\ran.li\PycharmProjects\\carotid_seg\\to_liran\\checkpoint_carotid\\train_bayesian_aug')
    results = model.evaluate(X_test2, y_test2, batch_size=batch_size)
    print('test loss, test acc:', results)
    for i in range(40):
        output = model.predict(X_test[:,:,:,:], batch_size=batch_size)
        output = tf.math.softmax(output)
        output = output.numpy()
        adict = {}
        if i == 0:
            var_output = np.zeros(output.shape)
            var_output = np.expand_dims(var_output, axis=4)
            var_output = np.repeat(var_output, 40, axis=4)
        var_output[:, :, :, :, i] = output
    output_mean = np.mean(var_output, axis=4)
    output_var = np.var(var_output, axis=4)
    adict = {}
    adict['seg_mean'] = output_mean
    adict['seg_var'] = output_var

    for i in range(40):
        output = model.predict(X_test2[:,:,:,:], batch_size=batch_size)
        output = tf.math.softmax(output)
        output = output.numpy()
        adict2 = {}
        if i == 0:
            var_output = np.zeros(output.shape)
            var_output = np.expand_dims(var_output, axis=4)
            var_output = np.repeat(var_output, 40, axis=4)
        var_output[:, :, :, :, i] = output
    output_mean = np.mean(var_output, axis=4)
    output_var = np.var(var_output, axis=4)
    adict = {}
    adict['seg_mean'] = output_mean
    adict['seg_var'] = output_var
    sio.savemat(
        'C:\\Users\\ran.li\\PycharmProjects\\carotid_seg\\matlab\\test_canf_1229_best.mat', adict)