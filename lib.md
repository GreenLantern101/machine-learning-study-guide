## Libraries

### ML
- scikit-learn
- AirBnB's [aerosolve](https://github.com/airbnb/aerosolve)

### Neural networks/deep-learning
- [Keras](https://github.com/fchollet/keras) (high level)
  - [Theano vs. TensorFlow backend](https://www.quora.com/Do-you-recommend-using-Theano-or-Tensor-Flow-as-Keras-backend)
- [TFLearn](https://github.com/tflearn/tflearn) - high-level API for deep-learning w/ TensorFlow
- Google's [Sonnet](https://github.com/deepmind/sonnet) - Tensorflow-based neural network library
- [Lasagne](https://github.com/Lasagne/Lasagne) - Lightweight library for neural networks in Theano. Lower-level than Keras.~~
- [nolearn](https://github.com/dnouri/nolearn) - scikit-learn compatible neural network library, wrapper around Lasagne
- ~~Microsoft's [CNTK](https://github.com/Microsoft/CNTK)~~
- ~~Facebook's [Torch](https://github.com/torch/torch7) - not as popular, written in Lua and C.~~
- ~~UC Berkeley's [Caffe](https://github.com/BVLC/caffe) -  ported AlexNet? implementation of fast convolutional nets to C and C++.~~

### Low-level
- Google's [TensorFlow](https://github.com/tensorflow/tensorflow) - created to replace Theano, but a bit less performant for RNNs
- [Theano](https://github.com/Theano/Theano) - very verbose, works with computational graphs

### Notes
- Most promising:
  - Keras on top of Theano/Tensorflow
  - nolearn on Lasagne on Theano
- Keras is [much less LOC](https://gist.github.com/cburgdorf/e2fb46e5ad61ed7b9a29029c5cc30134) than TensorFlow.
- Comparing Caffe, Torch, TensorFlow, Theano [here](https://www.youtube.com/watch?v=Qynt-TxAPOs)
