# WaveletDeconv
Neural network layer code written using Keras to implement wavelet deconvolutions.

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
