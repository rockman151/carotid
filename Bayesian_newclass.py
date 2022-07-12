from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
import tensorflow_probability as tfp
tf.keras.backend.set_image_data_format('channels_last')
tf.config.experimental.list_physical_devices('GPU')
import scipy.io as sio
########### cnn parameters
nb_ch=5
output_ch = 5
nb_fils=64
nb_mc=10
weight_decay=0.05
batch_size=2
_smooth = 1
img_size = 128
dr = 0.1 #dropout
filter_size = (3, 3)

class Conv_Bn_ReLU(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, name, dilation_rate=(1, 1)):
        super(Conv_Bn_ReLU, self).__init__(name=name)
        self.blocks = keras.Sequential()
        self.blocks.add(keras.layers.Conv2D(filters, kernel_size, strides, padding='same', dilation_rate=dilation_rate,kernel_initializer='glorot_normal'))
        self.blocks.add(keras.layers.BatchNormalization())
        if name.find('relu') != -1:
            self.blocks.add(keras.layers.ReLU())

    def __call__(self, inputs, **kwargs):
        output = self.blocks(inputs)

        return output

class Trans_Conv_Bn_ReLU(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, name, dilation_rate=(1, 1)):
        super(Trans_Conv_Bn_ReLU, self).__init__(name=name)
        self.blocks = keras.Sequential()
        self.blocks.add(keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same', dilation_rate=dilation_rate,kernel_initializer='glorot_normal'))
        self.blocks.add(keras.layers.BatchNormalization())
        if name.find('relu') != -1:
            self.blocks.add(keras.layers.ReLU())

    def __call__(self, inputs, **kwargs):
        output = self.blocks(inputs)

        return output

class Conv_Bn_ReLU_Prob(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, name, dilation_rate=(1, 1)):
        super(Conv_Bn_ReLU_Prob, self).__init__(name=name)
        self.blocks = keras.Sequential()
        self.blocks.add(tfp.layers.Convolution2DFlipout(filters, kernel_size, strides, padding='same', dilation_rate=dilation_rate))
        self.blocks.add(keras.layers.BatchNormalization())
        if name.find('relu') != -1:
            self.blocks.add(keras.layers.ReLU())

    def __call__(self, inputs, **kwargs):
        output = self.blocks(inputs)

        return output

def Network(input_size):
    input_x = tf.keras.Input(shape=input_size, batch_size=batch_size)
    x1 = Conv_Bn_ReLU(nb_fils, filter_size, (1, 1), name='layer1_conv_bn_relu')(input_x)
    x1p = Conv_Bn_ReLU(nb_fils, filter_size, (1, 1), name='layer2_conv_bn_relu')(x1)
    x2 = Conv_Bn_ReLU(nb_fils, filter_size, (2, 2), name='layer3_conv_bn_relu')(x1p)
    x2p = Conv_Bn_ReLU(nb_fils*2, filter_size, (1, 1), name='layer4_conv_bn_relu')(x2)
    x2d = Conv_Bn_ReLU(nb_fils*2, filter_size, (1, 1), name='layer5_conv_bn_relu')(x2p)
    x3 = Conv_Bn_ReLU(nb_fils*2, filter_size, (2, 2), name='layer6_conv_bn_relu')(x2d)
    x3p = Conv_Bn_ReLU(nb_fils*4, filter_size, (1, 1), name='layer7_conv_bn_relu')(x3)
    x3d = Conv_Bn_ReLU(nb_fils*4, filter_size, (1, 1), name='layer8_conv_bn_relu')(x3p)
    x4 = Conv_Bn_ReLU(nb_fils * 4, filter_size, (2, 2), name='layer9_conv_bn_relu')(x3d)
    x4p = Conv_Bn_ReLU(nb_fils * 8, filter_size, (1, 1), name='layer10_conv_bn_relu')(x4)
    x4d = Conv_Bn_ReLU(nb_fils * 8, filter_size, (1, 1), name='layer11_conv_bn_relu')(x4p)
    x5 = Conv_Bn_ReLU(nb_fils * 8, filter_size, (2, 2), name='layer12_conv_bn_relu')(x4d)
    x5p = Conv_Bn_ReLU(nb_fils * 16, filter_size, (1, 1), name='layer13_conv_bn_relu')(x5)
    x5d = Conv_Bn_ReLU(nb_fils * 16, filter_size, (1, 1), name='layer14_conv_bn_relu')(x5p)
    d4 = Trans_Conv_Bn_ReLU(nb_fils * 8, filter_size, (2, 2), name='layer15_deconv_bn_relu')(x5d)
    d4c = Conv_Bn_ReLU(nb_fils * 8, filter_size, (1, 1), name='layer16_conv_bn_relu')(tf.concat([x4d, d4], 3))
    d4d = Conv_Bn_ReLU_Prob(nb_fils * 8, filter_size, (1, 1), name='layer17_conv_bn_relu_prob')(d4c)
    d3 = Trans_Conv_Bn_ReLU(nb_fils * 4, filter_size, (2, 2), name='layer18_deconv_bn_relu')(d4d)
    d3c = Conv_Bn_ReLU(nb_fils * 4, filter_size, (1, 1), name='layer19_conv_bn_relu')(tf.concat([x3d, d3], 3))
    d3d = Conv_Bn_ReLU_Prob(nb_fils * 4, filter_size, (1, 1), name='layer20_conv_bn_relu_prob')(d3c)
    d2 = Trans_Conv_Bn_ReLU(nb_fils * 2, filter_size, (2, 2), name='layer21_deconv_bn_relu')(d3d)
    d2c = Conv_Bn_ReLU(nb_fils, filter_size, (1, 1), name='layer22_conv_bn_relu')(tf.concat([x2d, d2], 3))
    d2d = Conv_Bn_ReLU_Prob(nb_fils, filter_size, (1, 1), name='layer23_conv_bn_relu_prob')(d2c)
    d1 = Trans_Conv_Bn_ReLU(nb_fils, filter_size, (2, 2), name='layer24_deconv_bn_relu')(d2d)
    d1c = Conv_Bn_ReLU(nb_fils, filter_size, (1, 1), name='layer25_conv_bn_relu')(tf.concat([x1p, d1], 3))
    d1d = Conv_Bn_ReLU_Prob(nb_fils, filter_size, (1, 1), name='layer26_conv_bn_relu_prob')(d1c)
    d0c = Conv_Bn_ReLU(nb_fils, filter_size, (1, 1), name='layer27_conv_bn_relu')(d1d)
    output = Conv_Bn_ReLU(output_ch, 1, (1, 1), name='layer28_conv_bn')(d0c)
    model = tf.keras.Model(inputs=input_x, outputs=output)
    return model
def metrics_RH(y_true, y_pred):
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

def load_data(path,para_name):
    data = sio.loadmat(path)# load . mat data
    data = np.array(data[para_name])
    return data

model = Network(input_size=(img_size,img_size,nb_ch))
opt=tf.keras.optimizers.Adam(
    learning_rate=0.0005, beta_1=0.9, beta_2=0.9999, epsilon=1e-07, amsgrad=False,
    name='Adam'
)
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
    KLD = sum(model.losses)
    loss = dice_loss + mc_x + KLD
    return loss
model.compile(loss = loss_fn, optimizer = opt, metrics = [[metrics_RH]],experimental_run_tf_function=False)

Train_all = load_data(path = '\Users\example_train_BNN.mat',
                      para_name = 'img')
Label  = load_data(path = '\Users\example_label_BNN.mat',
                   para_name= 'seg')

k_X_train, k_X_val, k_y_train, k_y_val = train_test_split(Train_all, Label, test_size=5/30)
log_dir = '\Users\tensorboard'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
for ite in range(10):
    if ite>0:
        model.load_weights('\Users\checkpoint_carotid\train_bayesian_best_carotid2')
    history = model.fit(k_X_train,k_y_train,batch_size=batch_size, epochs=10,validation_data=(k_X_val, k_y_val), verbose=1, shuffle=True,callbacks=[tensorboard_callback])
    model.save_weights('\Users\checkpoint_carotid\train_bayesian_best_carotid2')


