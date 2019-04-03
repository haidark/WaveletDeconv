from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.utils import conv_utils
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

class Pos(constraints.Constraint):
    '''Constrain the weights to be strictly positive
    '''
    def __call__(self, p):
        p *= K.cast(p > 0., K.floatx())
        return p

class WaveletDeconvolution(Layer):
    '''
    Deconvolutions of 1D signals using wavelets
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`  as a
    (tuple of integers, e.g. (10, 128) for sequences
    of 10 vectors with dimension 128).
    
    # Example
    ```python
        # apply a set of 5 wavelet deconv widthss to a sequence of 32 vectors with 10 timesteps
        model = Sequential()
        model.add(WaveletDeconvolution(5, padding='same', input_shape=(32, 10)))
        # now model.output_shape == (None, 32, 10, 5)
        # add a new conv2d on top
        model.add(Convolution2D(64, 3, 3, padding='same'))
        # now model.output_shape == (None, 64, 10, 5)
    ```
    # Arguments
        nb_widths: Number of wavelet kernels to use
            (dimensionality of the output).
        kernel_length: The length of the wavelet kernels            
        init: Locked to didactic set of widths ([1, 2, 4, 8, 16, ...]) 
            name of initialization function for the weights of the layer
            (see [initializers](../initializers.md)),
            or alternatively, a function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        activation: name of activation function to use
            ( or alternatively, an elementwise function.)
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the bias vector
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix
        bias_constraint: Constraint function applied to the bias vector
    
    # Input shape
        if data_format is 'channels_first' then
            3D tensor with shape: `(batch_samples, input_dim, steps)`.
        if data_format is 'channels_last' then
            3D tensor with shape: `(batch_samples, steps, input_dim)`.
        
    # Output shape
        if data_format is 'channels_first' then
            4D tensor with shape: `(batch_samples, input_dim, new_steps, nb_widths)`.
            `steps` value might have changed due to padding.
        if data_format is 'channels_last' then
            4D tensor with shape: `(batch_samples, new_steps, nb_widths, input_dim)`.
            `steps` value might have changed due to padding.
    '''
    
    def __init__(self, nb_widths, kernel_length=100,
                 init='uniform', activation='linear', weights=None,
                 padding='same', strides=1, data_format='channels_last', use_bias=True,
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 input_shape=None, **kwargs):

        if padding.lower() not in {'valid', 'same'}:
            raise Exception('Invalid border mode for WaveletDeconvolution:', padding)
        if data_format.lower() not in {'channels_first', 'channels_last'}:
            raise Exception('Invalid data format for WaveletDeconvolution:', data_format)
        self.nb_widths = nb_widths
        self.kernel_length = kernel_length
        self.init = self.didactic #initializers.get(init, data_format='channels_first')
        self.activation = activations.get(activation)
        self.padding = padding
        self.strides = strides

        self.subsample = (strides, 1)

        self.data_format = data_format.lower()

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = Pos()
        self.bias_constraint = constraints.get(bias_constraint)

        self.use_bias = use_bias
        self.initial_weights = weights
        super(WaveletDeconvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        # get dimension and length of input
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]
            self.input_length = input_shape[2]
        else:
            self.input_dim = input_shape[2]
            self.input_length = input_shape[1]
        # initialize and define wavelet widths
        self.W_shape = (self.nb_widths)
        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]
        
        self.regularizers = []

        if self.kernel_regularizer:
            self.kernel_regularizer.set_param(self.W)
            self.regularizers.append(self.kernel_regularizer)

        if self.use_bias and self.bias_regularizer:
            self.bias_regularizer.set_param(self.b)
            self.regularizers.append(self.bias_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.kernel_constraint:
            self.constraints[self.W] = self.kernel_constraint
        if self.use_bias and self.bias_constraint:
            self.constraints[self.b] = self.bias_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(WaveletDeconvolution, self).build(input_shape)
        
    def call(self, x, mask=None):
        # shape of x is (batches, input_dim, input_len) if 'channels_first'
        # shape of x is (batches, input_len, input_dim) if 'channels_last'
        # we reshape x to channels first for computation
        if self.data_format == 'channels_last':
            x = tf.transpose(x, (0, 2, 1))

        #x = K.expand_dims(x, 2)  # add a dummy dimension for # rows in "image", now shape = (batches, input_dim, input_len, 1)
        
        # build the kernels to convolve each input signal with
        kernel_length = self.kernel_length
        T = (np.arange(0,kernel_length) - (kernel_length-1.0)/2).astype('float32')
        T2 = T**2
        # helper function to generate wavelet kernel for a given width
        # this generates the Mexican hat or Ricker wavelet. Can be replaced with other wavelet functions.
        def gen_kernel(w):
            w2 = w**2
            B = (3 * w)**0.5
            A = (2 / (B * (np.pi**0.25)))
            mod = (1 - (T2)/(w2))
            gauss = K.exp(-(T2) / (2 * (w2)))
            kern = A * mod * gauss
            kern = K.reshape(kern, (kernel_length, 1))
            return kern
        wav_kernels = []
        for i in range(self.nb_widths):
            kernel = gen_kernel(self.W[i])
            wav_kernels.append(kernel)
        wav_kernels = tf.stack(wav_kernels, axis=0)
        # kernel, _ = tf.map_fn(fn=gen_kernel, elems=self.W)
        wav_kernels = K.expand_dims(wav_kernels, 0)
        wav_kernels = tf.transpose(wav_kernels,(0, 2, 3, 1))               

        # reshape input so number of dimensions is first (before batch dim)
        x = tf.transpose(x, (1, 0, 2))
        def gen_conv(x_slice):
            x_slice = K.expand_dims(x_slice,1) # shape (num_batches, 1, input_length)
            x_slice = K.expand_dims(x_slice,2) # shape (num_batches, 1, 1, input_length)
            return K.conv2d(x_slice, wav_kernels, strides=self.subsample, padding=self.padding, data_format='channels_first')
        outputs = []
        for i in range(self.input_dim):
            output = gen_conv(x[i,:,:])
            outputs.append(output)
        outputs = tf.stack(outputs, axis=0)
        # output, _ = tf.map_fn(fn=gen_conv, elems=x)
        outputs = K.squeeze(outputs, 3)
        outputs = tf.transpose(outputs, (1, 0, 3, 2))
        if self.data_format == 'channels_last':
            outputs = tf.transpose(outputs,(0, 2, 3, 1))
        return outputs
                
    def compute_output_shape(self, input_shape):
        out_length = conv_utils.conv_output_length(input_shape[2], 
                                                   self.kernel_length, 
                                                   self.padding, 
                                                   self.strides)        
        return (input_shape[0], self.input_dim, out_length, self.nb_widths)
    
    def get_config(self):
        config = {'nb_widths': self.nb_widths,
                  'kernel_length': self.kernel_length,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format,
                  'kernel_regularizer': self.kernel_regularizer.get_config() if self.kernel_regularizer else None,
                  'bias_regularizer': self.bias_regularizer.get_config() if self.bias_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'kernel_constraint': self.kernel_constraint.get_config() if self.kernel_constraint else None,
                  'bias_constraint': self.bias_constraint.get_config() if self.bias_constraint else None,
                  'use_bias': self.use_bias}
        base_config = super(WaveletDeconvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   
    
    def didactic(self, shape, name=None):
        x = 2**np.arange(shape).astype('float32')
        return K.variable(value=x, name=name)

if __name__ == '__main__':
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D
    
    model = Sequential()
    model.add(WaveletDeconvolution(4, kernel_length=30, input_shape=(2, 100), padding='same', data_format='channels_first'))
    model.compile(optimizer='sgd')

    print('tester code to visualize outputs')
    ### tester code to visualize outputs        
    tester = np.random.random((1, 2, 100)).astype('float32')
    z = model.predict(tester)
    print(z.shape)
    with K.get_session().as_default():
        for i in range(4):
            plt.figure(figsize=(10,4))            
            plt.subplot(121)
            plt.plot(np.squeeze(z[0,0,:,i]), 'k')
            plt.plot(np.squeeze(tester[:,0,:]), 'b')            
            plt.title('Channel 1 filtered signal (black). Width=%.2f' % model.layers[0].weights[0][i].eval())
            plt.subplot(122)
            plt.plot(np.squeeze(z[0,1,:,i]), 'r')
            plt.plot(np.squeeze(tester[:,1,:]), 'g')
            plt.title('Channel 2 filtered signal (red). Width=%.2f' % model.layers[0].weights[0][i].eval())
            plt.show()