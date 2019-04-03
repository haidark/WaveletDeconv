# WaveletDeconv
Neural network layer code written using Keras to implement Wavelet Deconvolutions from the paper:

Khan, Haidar, and Bulent Yener. "Learning filter widths of spectral decompositions with wavelets." Advances in Neural Information Processing Systems. 2018.

Requires Keras with a Tensorflow backend in addition to standard packages such as `numpy`, `matplotlib`, `scipy`, and `h5py`.

Run `testWD.py` to verify model saving, model loading, and proper functionality.

Deconvolutions of 1D signals using wavelets of different scales/widths. For a full description of the wavelet deconolution method, see our [paper](http://papers.nips.cc/paper/7711-learning-filter-widths-of-spectral-decompositions-with-wavelets.pdf)

    
### Code Example
```python
    # apply a set of 5 wavelet deconv widthss to a sequence of 32 vectors with 10 timesteps
    model = Sequential()
    model.add(WaveletDeconvolution(5, kernel_length=200, padding='same', input_shape=(32, 10), data_format='channels_first'))
    # now model.output_shape == (None, 32, 10, 5)
    # add a conv2d on top
    model.add(Convolution2D(64, 3, 3, padding='same'))
    # now model.output_shape == (None, 64, 10, 5)
```
