from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.utils import conv_utils
from keras.engine.topology import Layer
import numpy as np
from matplotlib import pyplot as plt
import theano

class Pos(constraints.Constraint):
    '''Constrain the weights to be strictly positive
    '''
    def __call__(self, p):
        p *= K.cast(p > 0., K.floatx())
        return p

class WaveletDeconvolution(Layer):
    '''
    Deconvolutions of 1D signals using wavelets of different sizes
    When using this layer as the first layer in a model,
    either provide the keyword argument `input_dim`
    (int, e.g. 128 for sequences of 128-dimensional vectors),
    or `input_shape` (tuple of integers, e.g. (10, 128) for sequences
    of 10 vectors of 128-dimensional vectors).
    
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
        nb_widths: Number of convolution kernels to use
            (dimensionality of the output).
        filter_length: The length of the wavelet kernel windows            
        init: Locked to didactic set of widths ([1, 2, 4, 8, 16, ...]) 
            name of initialization function for the weights of the layer
            (see [initializers](../initializers.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        padding: 'valid' or 'same'.
        subsample_length: factor by which to subsample output.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: Locked to strictly positive weights as widths must be positive
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
    
    # Input shape
        3D tensor with shape: `(batch_samples, input_dim, steps)`.
        
    # Output shape
        4D tensor with shape: `(batch_samples, input_dim, new_steps, nb_widths)`.
        `steps` value might have changed due to padding.
    
    '''
    
    def __init__(self, nb_widths, filter_length=100,
                 init='uniform', activation='linear', weights=None,
                 padding='same', subsample_length=1,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_length=None, input_dim=None, **kwargs):

        if padding not in {'valid', 'same'}:
            raise Exception('Invalid border mode for WaveletDeconvolution:', padding)
        self.nb_widths = nb_widths
        self.filter_length = filter_length
        self.init = self.didactic #initializers.get(init, data_format='channels_first')
        self.activation = activations.get(activation)
        self.padding = padding
        self.subsample_length = subsample_length

        self.subsample = (subsample_length, 1)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = Pos()
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_length = input_length
        self.input_dim = input_dim
        super(WaveletDeconvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[1]
        self.input_length = input_shape[2]
        self.W_shape = (self.nb_widths)
        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]
        
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(WaveletDeconvolution, self).build(input_shape)
        
    def call(self, x, mask=None):
        # shape of x is (batches, input_dim, input_len)
        #x = K.expand_dims(x, 2)  # add a dummy dimension for # rows in "image", now shape = (batches, input_dim, input_len, 1)
        
        # build the kernels to convolve each input signal with
        filter_length = self.filter_length
        X = (np.arange(0,filter_length) - (filter_length-1.0)/2).astype('float32')
        X2 = X**2
        def gen_kernel(w):
            w2 = w**2
            B = (3 * w)**0.5
            A = (2 / (B * (np.pi**0.25)))
            mod = (1 - (X2)/(w2))
            gauss = K.exp(-(X2) / (2 * (w2)))
            kern = A * mod * gauss
            kern = K.reshape(kern, (filter_length, 1))
            return kern
        kernel, _ = theano.scan(fn=gen_kernel, sequences=[self.W])
        kernel = K.expand_dims(kernel, 0)
        kernel = kernel.transpose((0, 2, 3, 1))
               
        # kernel = None     
        # for i in range(self.nb_widths):
        #     w = self.W[i]
        #     w2 = w**2
        #     B = (3 * w)**0.5
        #     A = (2 / (B * (np.pi**0.25)))
        #     mod = (1 - (X2)/(w2))
        #     gauss = K.exp(-(X2) / (2 * (w2)))
        #     kern = A * mod * gauss
        #     kern = K.reshape(kern, (filter_length, 1))
        #     wav_kernel = K.transpose(kern)              # shape is now (1, filter_length)
        #     wav_kernel = K.expand_dims(wav_kernel, 0)   # shape is now (1, 1, filter_length)
        #     wav_kernel = K.expand_dims(wav_kernel, 1)   # shape is now (1, 1, 1, filter_length)
        #     if kernel is None:
        #         kernel = wav_kernel
        #     else:
        #         kernel = K.concatenate([kernel, wav_kernel], axis=0)
        # kernel = kernel.transpose((1, 3, 2, 0)) # TF style shape = (1, filter_length, 1, nb_widths)

        
        
        # # compute the convolution
        # output = None
        # for i in range(self.input_dim):
        #     x_slice = x[:,i,:,:]
        #     x_slice = K.expand_dims(x_slice,1) # shape (num_batches, 1, 1, input_length)
        #     output_slice = K.conv2d(x_slice, kernel, strides=self.subsample, padding=self.padding, data_format='channels_first')
        #     if output is None:
        #         output = output_slice
        #     else:
        #         output = K.concatenate([output, output_slice], axis=2)
        # output = output.transpose((0, 2, 3, 1)) # shape (num_batches, input_dim, input_length, nb_widths)
        # output = self.activation(output)
        
        # reshape input so number of dimensions is first
        x = x.transpose((1, 0, 2))
        def gen_conv(x_slice):
            x_slice = K.expand_dims(x_slice,1) # shape (num_batches, 1, input_length)
            x_slice = K.expand_dims(x_slice,2) # shape (num_batches, 1, 1, input_length)
            return K.conv2d(x_slice, kernel, strides=self.subsample, padding=self.padding, data_format='channels_first')
        output, _ = theano.scan(fn=gen_conv, sequences=[x])
        output = K.squeeze(output, 3)
        output = output.transpose((1, 0, 3, 2))
                 
        # ### tester code to visualize outputs
        # tester = np.random.random((1, 2, 100)).astype('float32')
        # z = output.eval({x : tester})
        # print z.shape
#         for i in range(self.nb_widths):
#             plt.figure(figsize=(10,4))
               
#             plt.subplot(121)
#             plt.plot(np.squeeze(z[0,0,:,i]), 'k')
#             plt.plot(np.squeeze(tester[:,0,:]), 'b')
#             plt.title('Width=%f' % self.W[i].eval())
#             plt.subplot(122)
#             plt.plot(np.squeeze(z[0,1,:,i]), 'r')
#             plt.plot(np.squeeze(tester[:,1,:]), 'g')
#             plt.title('Width=%f' % self.W[i].eval())
#             plt.show()
            
        return output
                
    def compute_output_shape(self, input_shape):
        out_length = conv_utils.conv_output_length(input_shape[2], 
                                                   self.filter_length, 
                                                   self.padding, 
                                                   self.subsample_length)        
        return (input_shape[0], self.input_dim, out_length, self.nb_widths)
    
    def get_config(self):
        config = {'nb_widths': self.nb_widths,
                  'filter_length': self.filter_length,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'padding': self.padding,
                  'subsample_length': self.subsample_length,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(WaveletDeconvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   
    
    def didactic(self, shape, name=None):
        x = 2**np.arange(shape).astype('float32')
        return K.variable(value=x, name=name)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
 
modelWD = Sequential()
modelWD.add(WaveletDeconvolution(4, filter_length=30, input_shape=(2, 100), padding='same'))

