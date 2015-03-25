from __future__ import division

__author__ = 'isaac'

import operator
import cPickle
import time
import copy
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.ifelse import ifelse
import theano.printing
import tools
from math import ceil
import os

theano.config.exception_verbosity = 'high'
theano.config.floatX = 'float32'
theano_print = []


def _debug(string, var):
    global theano_print
    theano_print += theano.printing.Print("####### " + string + " #######")(var)


########################
# Activation Functions #
########################

def relu(x):
    """
    Used in conjunction with :class:`_NonLinearityLayer` to create a rectified
    linear unit layer.  The user does not instantiate this directly, but instead
    passes the class as the ``activation`` or ``weight_init`` argument to
    :class:`NN` either  when creating it or adding
    certain kinds of layers.

    :param float x: input to the rectified linear unit
    :return: 0 if x < 0, otherwise x
    :rtype: float
    """
    return x * (x > 0)
    # return T.maximum(x, 0)


def tanh(x):
    """
    Used in conjunction with :class:`_NonLinearityLayer` to create a hyperbolic
    tangent unit layer.  The user does not instantiate this directly, but instead
    passes the class as the ``activation`` or ``weight_init`` argument to
    :class:`NN` either  when creating it or adding
    certain kinds of layers.

    :param float x: input to the hyperbolic tangent unit
    :return: symbolic hyperbolic tangent function of x
    :rtype: float
    """
    return T.tanh(x)


def sigmoid(x):
    """
    Used in conjunction with :class:`_NonLinearityLayer` to create a sigmoid  unit
    layer.  The user does not instantiate this directly, but instead
    passes the class as the ``activation`` or ``weight_init`` argument to
    :class:`NN` either  when creating it or adding
    certain kinds of layers.

    :param float x: input to the sigmoid unit
    :return: symbolic logistic function of x
    :rtype: float
    """
    return theano.tensor.nnet.sigmoid(x)


##################
# Misc Functions #
##################

def _init_weights(layer_name, activation, weights_shape, bias_shape,
                  fan_in, fan_out, rng, use_bias):
    """
    Generates initial weights for :class:`_ConvPoolLayer` and
    :class:`_FullyConnectedLayer` depending on the activation function being
    used

    :param string layer_name: used to label Theano variables for debugging purposes
    :param activation: the activation function used; one of :func:`relu`, :func:`tanh`
        :func:`sigmoid`, or :class:`PReLULayer`
    :param tuple weights_shape: the shape of the weight array
    :param tuple bias_shape: the shape of the bias array
    :param int fan_in: the fan in to the units
    :param int fan_out: the fan out to the units
    :param rng: random number generator
    :param bool use_bias: ``True`` to generate bias weights; ``False`` not to.  (When
        Using batch normalization, bias is redundant and thus should not be used.)
    :return: Theano shared variable with initialized weights

    Notes:
    ------

    see:  http://www.reddit.com/r/MachineLearning/comments/29ctf7/how_to_initialize_rectifier_linear_units_relu
    "I tend to initialise the weights from N(0, 0.01), and if that works I usually
    don't touch this parameter after that. I set the biases to 0.0 by default,
    sometimes to 0.1 or even 1.0 to avoid dead units (i.e. units that get stuck
    in the saturation zone and never recover because there is no gradient).

    "For very large layers, it is sometimes helpful to reduce the variance of the
    initial weights, so that their output is not too large. Then I sometimes try
    N(0, 0.001) instead. The same is true for very small layers (i.e. convolutional
    layers with small filters, so the total output is small), there I sometimes
    use N(0, 0.1).

    "What's important in the end is the relative magnitudes of the gradient updates
    and of the weights themselves. If the ratio between these becomes too large
    or too small, your network is not going to learn much. That's why it is
    important to initialise the weights to the right ranges: the magnitude of
    the gradient updates for a given layer depends on the magnitude of the
    weights in this layer, but also that of the weights in other layers of the
    network. In other words, changing the initialisation for a specific layer
    can also affect learning in other layers.

    "Hinton tends to recommend sampling the weights from the unit normal
    distribution times a scaling constant (such as 1e-2 or 1e-3) selected
    via some parameter search. In the original dropout paper he also seems
    to do the bias units this way for some networks, and use a constant of 1.0
    for other networks (see section F.5).

    see: https://plus.google.com/+EricBattenberg/posts/f3tPKjo7LFa
    "Andrej Karpathy:  I usually scale the weights with 1/sqrt(fanIn) because
    it makes sense to normalize the distribution of activations (as you mentioned),
    otherwise neurons with a lot of inputs will have much more diffuse outputs
    (higher std), but never properly played with this in detail and if it matters
    too much.
    """
    if activation in (relu, PReLULayer):
        w_values = np.asarray(
            rng.normal(
                loc=0.,
                # scale=.01,
                scale=1 / np.sqrt(fan_in),
                size=weights_shape),
            dtype=theano.config.floatX)

        # using this commented tanh code gets similar results as normal
        # with scale 1/sqrt(fan_in):
        # W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        # W_values = numpy.asarray(
        # rng.uniform(low=-W_bound, high=W_bound, size=weights_shape),
        # dtype=theano.config.floatX)

        b_values = np.ones(bias_shape,
                           dtype=theano.config.floatX) if use_bias else None

        # smaller biases don't seem to make a difference
        # b_values = numpy.ones(bias_shape, dtype=theano.config.floatX) / 10
    elif activation in (tanh, sigmoid):
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        w_values = np.asarray(
            rng.uniform(low=-w_bound, high=w_bound, size=weights_shape),
            dtype=theano.config.floatX)
        if activation == theano.tensor.nnet.sigmoid:
            w_values *= 4
        b_values = np.zeros(bias_shape,
                            dtype=theano.config.floatX) if use_bias else None
    else:
        raise Exception("specified activation not supported")

    w = theano.shared(value=w_values, name='W_' + layer_name, borrow=True)
    b = theano.shared(value=b_values, name='b_' + layer_name,
                      borrow=True) if use_bias else None
    return w, b


def _two_dimensional(inp):
    """
    Given an input layer transforms its output to 2 dimensions (or if it is
    already 2 dimensions leaves it be).  Used to prepare input for
    layers that only process two dimensional input such as
    :class:`_FullyConnectedLayer`, :class:`_LogisticRegression` and
    :class:`_MergeLayer`.

    :param inp: the input layer
    :return: (transformed input, shape of transformed input)
    """
    if len(inp.out_shape) == 4:
        return (inp.output.flatten(2),
                (inp.out_shape[0],
                 inp.out_shape[1] * inp.out_shape[2] * inp.out_shape[3]))
    elif len(inp.out_shape) == 2:
        return inp.output, inp.out_shape
    else:
        raise Exception('{} dimensional input not supported; input '
                        'must be 2 or 4 dimensions'.format(len(inp.out_shape)))


##########
# Layers #
##########

class _ConvPoolLayer(object):
    """
    A convolution and maxpooling layer.  (No nonlinearity is applied
    in this layer because when using :class:`BatchNormalizationLayer`, it must
    come between the matrix multiplies and the nonlinearity.  Instead a
    :class:`_NonLinearityLayer` or :class:`PReLULayer` layer can be applied
    after the :class:`_ConvPoolLayer`.)

    :type inp: theano.tensor.dtensor4
    :param inp: symbolic image tensor, of shape image_shape
    :type inp_shape: tuple or list of length 4
    :param inp_shape: (batch size, num input feature maps,
                         image height, image width)
    :type filter_shape: tuple or list of length 4
    :param filter_shape: (number of filters, num input feature maps,
        filter height, filter width).  "Usually the filters are odd size to have a
        central pixel. It depends on the calculation cost and the layer of the network
        that we are considering."
    :type pool_size: tuple or list of length 2
    :param pool_size: size of the pools
    :param pool_stride: stride between pools
    :param weight_init: activation function that will be applied
        after the :class:`ConvPoolLayer` (used
        to determine a weight initialization scheme--one of :func:`relu`,
        :func:`tanh`, :func:`sigmoid`, or :class:`PReLULayer`)
    :param use_bias: ``True`` to use bias; ``False`` not to.  (When
        using batch normalization, bias is redundant and thus should not be used.)
    :param rng: random number generator
    """

    def __init__(self, inp, inp_shape, filter_shape, pool_size, pool_stride,
                 weight_init, use_bias, rng):
        assert inp_shape[1] == filter_shape[1]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        # pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(pool_size))

        w, b = _init_weights('conv_pool', weight_init, filter_shape,
                             (filter_shape[0],), fan_in, fan_out, rng, use_bias)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=inp,
            filters=w,
            filter_shape=filter_shape,
            image_shape=inp_shape
        )
        conv_out_shape = ((inp_shape[2] - filter_shape[2] + 1),
                          (inp_shape[3] - filter_shape[3] + 1))

        # downsample each feature map individually, using maxpooling
        self.output = downsample.max_pool_2d(
            input=conv_out,
            ds=pool_size,
            st=pool_stride,
            ignore_border=True,
        )

        pool_shape = (int(ceil((conv_out_shape[0] - pool_size[0] + 1)
                               / pool_stride[0])),
                      int(ceil((conv_out_shape[1] - pool_size[1] + 1)
                               / pool_stride[1])))

        self.out_shape = (inp_shape[0], filter_shape[0],
                          pool_shape[0], pool_shape[1])

        self.params = [w]

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map width & height
        if use_bias:
            self.output += b.dimshuffle('x', 0, 'x', 'x')
            self.params += [b]

        print(('| ConvPoolLayer (inp_shape: {}, out_shape: {}, '
               'filter_shape: {}, pool_size: {}, pool_stride {}, use_bias: {})')
              .format(inp_shape, self.out_shape, filter_shape,
                      pool_size, pool_stride, use_bias))


class _LogisticRegression(object):
    """ Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.

    :type inp: theano.tensor.TensorType
    :param inp: symbolic variable that describes the input of the
                  architecture (one minibatch)

    :param y: symbolic variable for the labels

    :type n_in: int
    :param n_in: number of input units, the dimension of the space in
                 which the datapoints lie

    :type n_out: int
    :param n_out: number of output units, the dimension of the space in
                  which the labels lie

    """

    def __init__(self, inp, y, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        w = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W_logistic_regression',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b_logistic_regression',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(inp, w) + b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [w, b]

        self.negative_log_likelihood = -T.mean(
            T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            self.errors = T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

        print('| LogisticRegression (n_in: {}, n_out: {})'.format(n_in, n_out))


class _DropoutLayer(object):
    """
    See `the dropout paper <http://arxiv.org/pdf/1207.0580v1.pdf>`_.

    Randomly mask inputs with zeros with frequency ``rate`` while
    training, and scales inputs by ``1.0 - rate`` when not training
    so that aggregate signal sent to next layer will be roughly the
    same during training and inference.

    :param inp: input to this layer (output of previous layer)
    :param inp_shape: shape of input
    :param rate: rate at which to mask input values (between 0 and 1)
    :param train_flag: :attr:`NN.train_flag` (symbolic flag indicating
        whether training is happening)
    :param srng: :attr:`NN.snrg` (symbolic random number generator)

    Regarding Theano implementation, see:
    https://github.com/stencilman/deep_nets_iclr04/blob/master/lib/layer_blocks.py#L174
    https://groups.google.com/forum/#!topic/theano-users/u81jIHRLzUc
    https://blog.wtf.sg/2014/07/23/dropout-using-theano/
    """

    def __init__(self, inp, inp_shape, rate, train_flag, srng):
        assert 0 <= rate <= 1

        # create a mask in the shape of input with zeros at 'rate' and
        # ones at '1 - rate', so that when multiplied by input will
        # zero out elements at the rate 'rate'.
        # the cast in the following expression is important because:
        # int * float32 = float64 which pulls things off the gpu
        mask = T.cast(srng.binomial(n=1, p=1.0 - rate, size=inp_shape),
                      theano.config.floatX)
        off_gain = 1.0 - rate
        # if this is a training pass then multiply by mask to zero out
        # a portion of the inputs, if it is a prediction pass then scale
        # inputs down so that in aggregate they are sending as much signal
        # to the next layer as they would be if the mask was not switched off.
        # multiplying by T.ones_like seems to ensure that the two branches of
        # the ifelse output the same Types.  Otherwise theano throws an exception.
        self.output = inp * ifelse(train_flag, mask, off_gain * T.ones_like(mask))
        self.out_shape = inp_shape

        print('| DropoutLayer (inp_shape: {}, rate: {})'.format(inp_shape, rate))


class _BatchNormalizationLayer(object):
    """
    See the `batch normalization paper <http://arxiv.org/pdf/1502.03167v2.pdf>`_

    :param inp: input to this layer (output of previous layer)
    :param inp_shape: shape of input
    :param train_flag: :attr:`NN.train_flag` (symbolic flag indicating
    whether training is happening)
    :param rng: random number generator
    :param epsilon: epsilon from the paper (see link above)

    Regarding Theano implementation, see:
    https://github.com/takacsg84/Lasagne/blob/d5545988e6484d1db4bb54bcfa541ba62e898829/lasagne/layers/bn2.py
    https://gist.github.com/skaae/5faacedb9c5961136e82
    https://github.com/benanne/Lasagne/issues/141
    """

    def __init__(self, inp, inp_shape, train_flag, rng, epsilon=1e-6):
        if len(inp_shape) == 4:
            stat_shape = (1, inp_shape[1], 1, 1)
            broadcast = (True, False, True, True)
            mean_var_axis = (0, 2, 3)
        elif len(inp_shape) == 2:
            stat_shape = (1, inp_shape[1])
            broadcast = (True, False)
            mean_var_axis = 0
        else:
            raise Exception('input shape not supported; should be 2 or 4 dimensional')

        # for some reason theano gets confused if params are broadcastable during
        # parameter update calculations (i.e. of AdaDelta) so I make broadcastable
        # versions for use only here in output calculation, and use
        # non-broadcastable versions for self.params and self.updates
        gamma = theano.shared(
            value=np.asarray(rng.uniform(0.95, 1.05, stat_shape),
                             dtype=theano.config.floatX),
            name='gamma_bn',
            borrow=True)
        gamma_b = T.patternbroadcast(gamma, broadcast)
        beta = theano.shared(
            value=np.zeros(stat_shape, dtype=theano.config.floatX),
            name='beta_bn',
            borrow=True)
        beta_b = T.patternbroadcast(beta, broadcast)

        # using an exponential moving average for inference statistics instead
        # of calculating them layer by layer for all data.  Ideally the
        # inference statistics would be calculated with the final training
        # weights only, but since that seems like a hassle, I'm using a
        # pretty rapid exponential decay instead.  In practice I'm not sure
        # how much difference it makes, or whether it is more important to
        # get a big sample.  Especially as the weights stabilize towards the end
        # of training, a big sample may be more beneficial.  One easy thing
        # I could do is make the rate of exponential decay a function of the
        # number of batches so as weights change slows sample size increases.
        # It seems like it probably isn't worth being too precise about this given
        # that the weights of the network are training on batch statistics anyway.
        ema_means = theano.shared(
            value=np.zeros(stat_shape, dtype=theano.config.floatX),
            name='ema_means_bn',
            borrow=True)
        ema_vars = theano.shared(
            value=np.zeros(stat_shape, dtype=theano.config.floatX),
            name='ema_vars_bn',
            borrow=True)

        # calculate the training batch normalization
        batch_means = T.mean(inp, mean_var_axis, keepdims=True)
        batch_vars = T.sqrt(T.var(inp, mean_var_axis, keepdims=True) + epsilon)
        batch_norm_x = (inp - batch_means) / batch_vars

        # calculate the inference normalization
        inf_means = ema_means * 0.65 + batch_means * 0.35
        inf_vars = ema_vars * 0.65 + batch_vars * 0.35
        inf_means_b = T.patternbroadcast(inf_means, broadcast)
        inf_vars_b = T.patternbroadcast(inf_vars, broadcast)
        inf_norm_x = (inp - inf_means_b) / inf_vars_b

        norm_x = ifelse(train_flag, batch_norm_x, inf_norm_x)

        self.output = gamma_b * norm_x + beta_b
        self.out_shape = inp_shape
        self.params = [gamma, beta]

        self.updates = [(ema_means, inf_means),
                        (ema_vars, inf_vars)]

        print('| BatchNormalizationLayer (inp_shape: {})'.format(inp_shape))


class _DataLayer(object):
    """
    Inputs a particular data channel into the network.

    :param inp: input to this layer (output of previous layer)
    :param inp_shape: shape of input
    :param channel: the name of the channel to input into the network.
        Channel names are determined by the preprocessor passed to
        :class:`NN`.
    """

    def __init__(self, inp, inp_shape, channel):
        self.output = inp.reshape(inp_shape)
        self.out_shape = inp_shape
        print('| DataLayer (inp_shape: {}, channel: {})'.format(inp_shape, channel))


class _MergeLayer(object):
    """
    Merge multiple layers (from different pathways) together.

    :param inputs: the input to be merged (the output from the pathways
        being merged); should be 2D tensor (4D Tensors should
        be flattened first).
    :param sizes: the sizes of each input (not including the batch size)
    :param batch_size: the batch size
    """

    def __init__(self, inputs, sizes, batch_size):
        self.out_shape = (batch_size, sum(sizes))
        self.output = T.concatenate(inputs, axis=1)

        print('| MergeLayer (inp_shapes: {}, out_shape: {})'.format(
            sizes, self.out_shape))


class _FullyConnectedLayer(object):
    """
    A matrix multiply and addition of biases.  (No nonlinearity is applied
    in this layer because when using :class:`_BatchNormalizationLayer`, it must
    come between the matrix multiplies and the nonlinearity.  Instead a
    :class:`_NonLinearityLayer` or :class:`PReLULayer` layer can be applied
    after the :class:`_FullyConnectedLayer`.)

    :param rng: random number generator
    :param inp: input to the :class:`_FullyConnectedLayer` (output of the
        previous layer)
    :param n_in: the size of the input (not including the batch size)
    :param n_out: the size of the output
    :param weight_init: activation function that will be applied
        after the :class:`_FullyConnectedLayer` (used
        to determine a weight initialization scheme--one of :func:`relu`,
        :func:`tanh`, :func:`sigmoid`, or :class:`PReLULayer`)
    :param use_bias: ``True`` to use bias; ``False`` not to.  (When
        using batch normalization, bias is redundant and thus should not be used.)
    :param w: ignore this; it is here to support autoencoders in the future
    :param b: ignore this; it is here to support autoencoders in the future
    """

    def __init__(self, rng, inp, n_in, n_out, batch_size, weight_init,
                 use_bias, w=None, b=None):
        if w is None:
            assert b is None
            w, b = _init_weights('hidden', weight_init, (n_in, n_out),
                                 (n_out,), n_in, n_out, rng, use_bias)

        self.out_shape = (batch_size, n_out)
        self.output = T.dot(inp, w)
        self.params = [w]

        if use_bias:
            self.output += b
            self.params += [b]

        print('| FullyConnectedLayer (n_in: {}, out_shape: {}, use_bias: {})'
              .format(n_in, self.out_shape, use_bias))


class _NonLinearityLayer(object):
    """ Applies a nonlinearity to the inputs

    :param inp: input to :class:`_NonLinearityLayer` (the output of the previous layer)
    :param inp_shape: shape of input
    :param nonlinearity: activation function that will be applied
        to the input--one of :func:`relu`,
        :func:`tanh`, :func:`sigmoid`, but not prelu which has its
        own layer type :class:`PReLULayer`)
    """

    def __init__(self, inp, inp_shape, nonlinearity):
        self.output = nonlinearity(inp)
        self.out_shape = inp_shape
        print('| NonLinearityLayer (nonlinearity: {})'.format(
            nonlinearity.__name__))


class PReLULayer(object):
    """
    Parametric Rectified Linear Units:  http://arxiv.org/pdf/1502.01852.pdf.
    The user does not instantiate this directly, but instead
    passes the class as the ``activation`` or ``weight_init`` argument to
    :class:`NN` either  when creating it or adding
    certain kinds of layers.

    Note: Don't use l1/l2 regularization with PReLU.  From the paper:
        "It is worth noticing that we do not use weight decay
        (l2 regularization) when updating a_i. A weight decay tends to push a_i
        to zero, and thus biases PReLU toward ReLU."
    """

    def __init__(self, inp, inp_shape):
        if len(inp_shape) == 4:
            a_shape = (1, inp_shape[1], 1, 1)
            broadcast = (True, False, True, True)
        elif len(inp_shape) == 2:
            a_shape = (1, inp_shape[1])
            broadcast = (True, False)
        else:
            raise Exception('input shape not supported; should be 2 or 4 dimensional')

        # for some reason theano gets confused if params are broadcastable during
        # parameter update calculations (of AdaDelta) so I make broadcastable
        # versions for use only here in output calculation, and use
        # non-broadcastable versions for self.params and self.updates
        a = theano.shared(
            value=0.25 * np.ones(a_shape, dtype=theano.config.floatX),
            name='a_PReLU',
            borrow=True)
        a_broad = T.patternbroadcast(a, broadcast)

        self.output = T.maximum(0, inp) + a_broad * T.minimum(0, inp)
        self.out_shape = inp_shape
        self.params = [a]
        print('| PReLULayer (inp_shape: {})'.format(inp_shape))


##################
# Learning Rates #
##################

class LearningRateAdjuster(object):
    """
    Base class for learning rate annealing: :class:`LearningRateDecay`,
    :class:`LearningRateSchedule`, and :class:`WackyLearningRateAnnealer`.

    :param float initial_learn_rate: the learning rate to start with on the
        first epoch
    """

    def __init__(self, initial_learn_rate):
        self.learning_rate = theano.shared(np.cast['float32'](initial_learn_rate))

    def _epoch_results(self, net, train_loss, valid_loss):
        """ see :meth:`_LearningRule.epoch_results` """
        return False, ''

    def _final_epochs(self):
        """ see :meth:`_LearningRule.final_epochs` """
        return ''


class LearningRateDecay(LearningRateAdjuster):
    """
    Decreases learning rate after each epoch according to formula:
    ``new_rate = initial_rate / (1 + epoch * decay)``

    :param float learning_rate: the initial learning rate
    :param float decay: the decay factor
    :param float min_learning_rate: the smallest ``learning_rate`` to which decay
        is applied; when ``learning_rate`` reaches ``min_learning_rate`` decay stops.
    """

    def __init__(self, learning_rate, decay, min_learning_rate=None):
        super(LearningRateDecay, self).__init__(learning_rate)
        self.initial_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay = decay
        self.epoch = 0

    def _epoch_results(self, net, train_loss, valid_loss):
        self.epoch += 1
        new_rate = self.initial_rate / (1 + self.epoch * self.decay)
        if (self.min_learning_rate is None or
            self.min_learning_rate < new_rate):
            self.learning_rate.set_value(np.cast['float32'](new_rate))
            return False, 'learning rate={:7.6f}, '.format(new_rate)
        else:
            return False, ''


class LearningRateSchedule(LearningRateAdjuster):
    """
    Sets the learning rate according to the given schedule.

    :param tuple schedule: list of pairs of epoch number and new learning rate.  For
        example, ``((0, .1), (200, .01), (300, .001))`` starts with a learning rate of
        .1, changes to a learning rate of .01 at epoch 200, and .001 at epoch 300.
    """

    def __init__(self, schedule):
        super(LearningRateSchedule, self).__init__(schedule[0][1])
        self.schedule = schedule

        # make sure the triggering epoch in each schedule
        # step comes after the previous one
        epochs = zip(*schedule)[0]
        for i in range(len(epochs) - 1):
            assert epochs[i] < epochs[i + 1]

        self.epoch = 0
        self.index = 1

    def _epoch_results(self, net, train_loss, valid_loss):
        self.epoch += 1
        if (self.index < len(self.schedule) and
                    self.schedule[self.index][0] == self.epoch):
            new_rate = self.schedule[self.index][1]
            self.learning_rate.set_value(np.cast['float32'](new_rate))
            self.index += 1
            if self.index == len(self.schedule):
                return False, 'final learning rate={:7.6f}, '.format(new_rate)
            else:
                return False, 'learning rate={:7.6f}, '.format(new_rate)

        if self.epoch == 1:
            return False, 'learning rate={:7.6f}, '.format(
                float(self.learning_rate.get_value()))
        else:
            return False, ''


class WackyLearningRateAnnealer(LearningRateAdjuster):
    """
    Decreases learning rate by factor of 10 after patience is depleted.
    Patience can be replenished by sufficient improvement in either training
    or validation loss.  Parameters of the network can optionally be reset
    to the parameters corresponding to the best training loss or to the
    best validation loss.

    :param float learning_rate: the initial learning rate
    :param float min_learning_rate: training stops upon reaching the min_learning_rate
    :param int patience: the number of epochs to train without sufficient improvement
        in training or validation loss before dropping the learning rate
    :param float train_improvement_threshold: how much training loss must improve over
        previous best training loss to trigger a reset of patience (if
        ``training_loss < best_training_loss * train_improvement_threshold``
        then patience is reset)
    :param float valid_improvement_threshold: how much validation loss must improve
        over previous best validation loss to trigger a reset of patience (if
        ``validation_loss < best_validation_loss * valid_improvement_threshold``
        then patience is reset)
    :param string reset_on_decay: one of `'training'`, `'validation'` or `None`; if
        `'training'` or `'validation'` then on learning rate decay network will
        be reset to the parameter values that correspond to the best training or
        validation scores.
    """

    def __init__(self, learning_rate, min_learning_rate,
                 patience=40, train_improvement_threshold=.995,
                 valid_improvement_threshold=0.99995,
                 reset_on_decay=None):
        super(WackyLearningRateAnnealer, self).__init__(learning_rate)
        self.min_learning_rate = tools.default(min_learning_rate, learning_rate / 100)
        self.patience = patience
        self.train_improvement_threshold = train_improvement_threshold
        self.valid_improvement_threshold = valid_improvement_threshold
        self.best_train_loss = np.inf
        self.reset_params = None
        self.best_valid_loss = np.inf
        self.best_valid_rate = learning_rate
        self.frust_train_loss = np.inf
        self.frust_valid_loss = np.inf
        self.frustration = 0
        self.first = True

        assert (reset_on_decay in ('training', 'validation', None))
        self.reset_on_decay = reset_on_decay

    def _epoch_results(self, net, train_loss, valid_loss):
        message = ''

        if self.first:
            message += 'learning rate={:7.6f}, '.format(
                float(self.learning_rate.get_value()))
            self.first = False

        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.best_valid_rate = self.learning_rate.get_value()
            if self.reset_on_decay == 'validation':
                self.reset_params = net._get_params()

        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            if self.reset_on_decay == 'training':
                self.reset_params = net._get_params()

        train_improved = train_loss < self.frust_train_loss * self.train_improvement_threshold
        valid_improved = valid_loss < self.frust_valid_loss * self.valid_improvement_threshold
        if train_improved or valid_improved:
            if train_improved:
                self.frust_train_loss = train_loss
            if valid_improved:
                self.frust_valid_loss = valid_loss
            self.frustration = 0
        else:
            if self.frustration >= self.patience:
                new_learn_rt = self.learning_rate.get_value() / 10.0
                if new_learn_rt < self.min_learning_rate:
                    return True, message + 'reached minimum learning rate, '
                else:
                    if self.reset_on_decay in ('training', 'validation'):
                        reset_mess = 'params=best {}'.format(
                            self.reset_on_decay)
                        net._set_params(self.reset_params)
                    else:
                        reset_mess = 'not resetting params'
                    self.learning_rate.set_value(np.cast['float32'](new_learn_rt))
                    message += 'learning rate={:7.6f} ({}), '.format(
                        new_learn_rt, reset_mess)
                    self.frustration = 0
            else:
                self.frustration += 1
                message += 'frustration={}, '.format(self.frustration)
        return False, message

    def _final_epochs(self):
        self.learning_rate.set_value(self.best_valid_rate)
        return 'set learning rate to {:7.6f}'.format(float(self.best_valid_rate))


##################
# Learning Rules #
##################

class LearningRule(object):
    """
    Base class for learning rules: :class:`StochasticGradientDescent`,
    :class:`Adam`, :class:`AdaDelta`, :class:`Momentum`.

    :param learning_rate: either a float (if using a constant learning rate)
        or a :class:`LearningRateAdjuster` (if using a learning rate that is
        adjusted during training)
    """

    def __init__(self, learning_rate):
        if isinstance(learning_rate, LearningRateAdjuster):
            self.learning_rate_adjuster = learning_rate
        else:
            self.learning_rate_adjuster = LearningRateAdjuster(learning_rate)

        self.learning_rate = self.learning_rate_adjuster.learning_rate

    def _epoch_results(self, net, train_loss, valid_loss):
        """
        This method is called *after* each epoch so the :class:`LearningRule`
        can make updates to its parameters based on the results of training.

        :param net: the :class:`NN` so the learning rule can get and set
            parameters if need be
        :param train_loss: the training loss for the epoch
        :param valid_loss: the validation loss for the epoch
        """
        return self.learning_rate_adjuster._epoch_results(
            net, train_loss, valid_loss)

    def _final_epochs(self):
        """
        Called before beginning final training on all datasets (training, validation,
        testing) so learning rule can set learning rate if need be.
        """
        return self.learning_rate_adjuster._final_epochs()

    def _get_updates(self, params, cost):
        raise NotImplementedError()


class StochasticGradientDescent(LearningRule):
    """
    Learn by `stochastic gradient descent <http://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_
    """

    def _get_updates(self, params, cost):
        grads = T.grad(cost, params)
        updates = [(param_i, param_i - self.learning_rate * grad_i)
                   for param_i, grad_i in zip(params, grads)]
        return updates


class Adam(LearningRule):
    """
    Learn by the `Adam optimization method <http://arxiv.org/pdf/1412.6980v4.pdf>`_

    Parameters are as specified in the paper above.

    ..  Regarding theano implementation, see:
        https://gist.github.com/skaae/ae7225263ca8806868cb
    """

    def __init__(self, learning_rate, b1=0.9,
                 b2=0.999, e=1e-8, lmbda=1 - 1e-8):
        super(Adam, self).__init__(learning_rate)
        # suggested learning_rate in paper is 0.001
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.lmbda = lmbda

    def _get_updates(self, params, cost):
        b1, b2, e, lmbda = self.b1, self.b2, self.e, self.lmbda
        alpha = self.learning_rate
        updates = []
        tm1 = theano.shared(np.float32(0))
        t = tm1 + 1.
        b1_t = b1 * lmbda ** tm1
        for p_tm1, g_t in zip(params, T.grad(cost, params)):
            m_tm1 = theano.shared(p_tm1.get_value() * 0.)
            v_tm1 = theano.shared(p_tm1.get_value() * 0.)

            m_t = b1_t * m_tm1 + (1 - b1_t) * g_t
            v_t = b2 * v_tm1 + (1 - b2) * g_t ** 2
            mhat_t = m_t / (1 - b1 ** t)
            vhat_t = v_t / (1 - b2 ** t)
            p_t = p_tm1 - alpha * mhat_t / (T.sqrt(vhat_t) + e)

            updates.append((m_tm1, m_t))
            updates.append((v_tm1, v_t))
            updates.append((p_tm1, p_t))
        updates.append((tm1, t))
        return updates


class AdaDelta(LearningRule):
    """
    Learn by the `AdaDelta optimization method <http://arxiv.org/pdf/1212.5701v1.pdf>`_

    Parameters are as specified in the paper above.
    """

    def __init__(self, rho, epsilon, learning_rate):
        super(AdaDelta, self).__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon

    def _get_updates(self, params, cost):
        grads = T.grad(cost, params)
        grad_accums = [theano.shared(np.zeros(param.get_value().shape,
                                              dtype=theano.config.floatX))
                       for param in params]
        delta_accums = [theano.shared(np.zeros(param.get_value().shape,
                                               dtype=theano.config.floatX))
                        for param in params]

        updates = []
        for param, grad, grad_acc, delta_acc in zip(params, grads, grad_accums,
                                                    delta_accums):
            grad_acc_new = self.rho * grad_acc + (1 - self.rho) * grad ** 2
            updates.append((grad_acc, grad_acc_new))

            update_mult = grad * (T.sqrt(delta_acc + self.epsilon) /
                                  T.sqrt(grad_acc_new + self.epsilon))
            param_new = param - self.learning_rate * update_mult
            updates.append((param, param_new))

            delta_acc_new = self.rho * delta_acc + (1 - self.rho) * update_mult ** 2
            updates.append((delta_acc, delta_acc_new))
        return updates


class Momentum(LearningRule):
    """
    Learn by `SGD with momentum <http://www.jmlr.org/proceedings/papers/v28/sutskever13.pdf>`_.

    :param float initial_momentum: trainings starts with this momentum
    :param float max_momentum: momentum is gradually increased until it reaches
        `max_momentum`
    """

    def __init__(self, initial_momentum, max_momentum, learning_rate):
        super(Momentum, self).__init__(learning_rate)
        assert (0 <= initial_momentum <= 1)

        self.momentum = theano.shared(np.cast['float32'](initial_momentum))
        self.max_momentum = max_momentum
        self.best_train_loss = np.inf

    def _get_updates(self, params, cost):
        updates = []
        for param in params:
            # getting the value and multiplying by 0.0 is a way to get
            # a zero array of the same shape
            old_velocity = theano.shared(param.get_value() * 0.0,
                                         broadcastable=param.broadcastable)
            # see https://groups.google.com/forum/#!searchin/theano-users/order$20of$20updates/theano-users/dnsrGNNGtic/qH9INySUv5gJ
            # it looks like theano calculates new values for all shared variables before
            # updating any of them.  So I think I'm getting this right.
            # I want to set param(t+1) = param(t) + velocity(t).
            # velocity(t) = momentum * velocity(t-1) + learning_rate * gradient(t).
            # In the code below, I need to use the velocity expression twice because
            # theano uses
            # all the parameters before updating them.  If I simply update param with
            # param + old_velocity then I'm getting the update for the previous
            # iteration.  To see this, think about what happens on the first
            # iteration.  If I update param with param + param_update, then after
            # the first iteration, param is still param, because param_update is
            # zero on the first iteration.  Thus I need to use the new velocity twice,
            # once to update the parameter, and once to save the new velocity to use
            # for the
            # next iteration.
            new_velocity = self.momentum * old_velocity - self.learning_rate * T.grad(
                cost, param)
            updates.append((param, param + new_velocity))
            updates.append((old_velocity, new_velocity))

            # below is the momentum rule suggested in
            # http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
            # which seems to differ from
            # http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
            # by getting rid of learning rate and replacing it with 1.0 - momentum.
            # Also, it doesn't seem to update param
            # on the first iteration, because param_update
            # is zero on the first iteration.  So it's always an iteration behind.
            # Is this related to nesterov momentum?
            # updates.append((param, param - learning_rate * param_update))
            # updates.append((param_update, momentum*param_update +
            # (1.0 - momentum)*T.grad(cost, param)))
        return updates

    def _epoch_results(self, net, train_loss, valid_loss):
        message = ''
        if train_loss > self.best_train_loss:
            new_mom = self.momentum.get_value() * 1.01
            if new_mom < self.max_momentum:
                self.momentum.set_value(np.cast['float32'](new_mom))
                message += 'momentum={:4.3f}, '.format(float(new_mom))
        else:
            self.best_train_loss = train_loss

        stop, sup_mess = super(Momentum, self)._epoch_results(net, train_loss,
                                                              valid_loss)
        message += sup_mess
        return stop, message


class NN(object):
    """a neural network to which you can add layers and subsequently train on data

    :param preprocessor:  a preprocessor for the data provides all the
        training, validation and test data to :class:`NN` during training
    :param string channel:  the initial channel to request from the preprocessor
        for the main layer pathway
    :param int num_classes: the number of classes
    :param int batch_size:  the number of observations in a batch
    :param rng: random number generator
    :param activation:  the default activation function to be used when
        no activation function is explicitly provided
    :param string name:  a name to use as a stem for saving network parameters during
        training
    :param string output_dir:  the directory in which to save network parameters
        during training

    Networks are constructed by calling the :meth:`add_*` methods in sequence
    to add processing layers.  For example::

        net.add_convolution(72, (7, 7), (2, 2))
        net.add_dropout()
        net.add_convolution(128, (5, 5), (2, 2))
        net.add_dropout()
        net.add_convolution(128, (3, 3), (2, 2))
        net.add_dropout()
        net.add_hidden(3072)
        net.add_dropout()
        net.add_hidden(3072)
        net.add_dropout()
        net.add_logistic()

    The above creates a network with three convolutional layers with 72, 128 and
    128 filter maps respectively, two hidden layers, each with 3072 units, dropout
    with a rate of 0.5 between each main layer and batch normalization (by default
    on each main layer).

    There are a few convenience :meth:`add_*` methods
    which are just combinations of other add methods:  :meth:`add_convolution`,
    :meth:`add_mlp` and :meth:`add_hidden`.

    There are a few methods for creating different processing pathways
    that can split from and rejoin the main network.  For example::

        net.add_convolution(72, (7, 7), (2, 2))
        net.add_dropout()
        net.add_convolution(128, (5, 5), (2, 2))
        net.add_dropout()
        net.merge_data_channel('shapes')
        net.add_hidden(3072)
        net.add_dropout()
        net.add_logistic()

    Here a new data channel called `'shapes'` was merged after the convolution.
    `'shapes'` is a channel provided by the preprocessor with the original image
    sizes.  (This can be useful where image sizes vary in meaningful ways; since
    that information is lost when uniformly resizing images to be fed into the neural
    network, it can be recovered by feeding in the size information separately
    after convolutions.)  In addition to simply merging a new data channel,
    it is also possible to split off a new pathway, apply transformations to it,
    and merge it back to the main pathway with :meth:`split_pathways`,
    :meth:`merge_pathways`, :meth:`new_pathway`.

    Once a neural network architecture has been built up, the network can be trained
    with :meth:`train`.  After training, inference can be done with :meth:`predict`,
    and confusion matrices can be generated with :meth:`get_confusion_matrices`
    to examine the kinds of errors the network is making.
    """

    def __init__(self, preprocessor, channel, num_classes, batch_size,
                 rng, activation=relu, name='net', output_dir=''):
        self.name = name

        # symbolic placeholder variables that are replaced when
        # theano functions are compiled
        # if any new symbolic variables are added they may also need to be
        # added to repeat_pathway_end() and make_givens()
        self.index = T.lscalar('nn_init_index')  # index to a [mini]batch
        self.y = T.ivector(
            'nn_init_y')  # the labels are presented as 1D vector of [int] labels
        self.x_symbols = {}
        self.train_flag = T.bscalar('train_flag')

        self.preprocessor = preprocessor

        self.batch_size = batch_size
        self.layers = [_DataLayer(
            self.x_symbols.setdefault(channel, T.matrix(channel)),
            [batch_size] + list(self.preprocessor.shape_for(channel)),
            channel)]
        self.rng = rng
        self.srng = T.shared_randomstreams.RandomStreams(rng.randint(2 ** 30))
        self.activation = activation
        self.num_classes = num_classes
        self.output_dir = output_dir

    def new_pathway(self, channel):
        """
        Creates a new pathway starting from a data channel.  (After adding
        layers specific to this pathway, if any, the new pathway
        must subsequently be merged with main pathway using
        :meth:`merge_pathways`.)

        :param string channel: name of the channel as output from preprocessor
        :return: :class:`NN` to which layers can be separately added
        """
        new = copy.copy(self)
        new.layers = [_DataLayer(
            self.x_symbols.setdefault(channel, T.matrix(channel)),
            [self.batch_size] + list(self.preprocessor.shape_for(channel)),
            channel)]
        return new

    def split_pathways(self, num=None):
        """
        Splits pathways off from the :class:`NN` object.  Split pathways
        can have different sequences of layers added and then be remerged using
        :meth:`merge_pathways`.

        :param int num: number of new pathways to split off from the original
                    pathway.  If num is None then split just one new pathway.

        :return:  If num is not None, returns a list of the new pathways
                  (not including the original pathway); otherwise returns a single
                  new pathway.

        TODO: this is broken in at least one way.  If there is a dropout
        or batch normalization layer in a split pathway but not in the main
        pathway (the pathway by which merge_pathways is called) then make_givens
        will look through self.layers and not find any dropout or batch normalization
        layers, and will thus not include the train_switch.  At least Theano would
        probably throw an exception in that case.
        """
        if num is None:
            new = copy.copy(self)
            new.layers = new.layers[:]
            return new
        else:
            pathways = [copy.copy(self) for _ in range(num)]
            for pathway in pathways:
                pathway.layers = pathway.layers[:]
            return pathways

    def merge_pathways(self, pathways):
        """
        Merge pathways.

        :param pathways: pathways to merge.  `pathways` can be
                         a single pathway or a list of pathways
        """
        pathways = [pathways] if isinstance(pathways, NN) else pathways
        inputs, shapes = [], []
        for pathway in [self] + pathways:
            inp, shape = _two_dimensional(pathway.layers[-1])
            inputs += [inp]
            shapes += [shape[1]]
        layer = _MergeLayer(
            inputs=inputs,
            sizes=shapes,
            batch_size=self.batch_size
        )
        self.layers.append(layer)

    def merge_data_channel(self, channel):
        """
        Creates a new pathway for processing `channel` data and merges it without
        adding any pathway specific layers.

        :param string channel: name of the channel as output from preprocessor
        """
        self.merge_pathways(self.new_pathway(channel))

    def add_conv_pool(self, num_filters, filter_shape, pool_shape,
                      pool_stride=None, weight_init=None, use_bias=True):
        """
        Adds a convolution and pooling layer to the network (without a
        nonlinearity or batch normalization; if those are desired they
        can be added separately, or the convenience method :meth:`add_convolution`
        can be used).

        :param int num_filters: number of filter maps to create
        :param tuple filter_shape: two dimensional shape of filters
        :param tuple pool_shape: two dimensional shape of pools
        :param tuple pool_stride: distance between pool starting points; if this is
            less than `pool_shape` then pools will be overlapping
        :param weight_init: activation function that will be applied to for
            the purposes of initializing weights (this method will not apply
            the activation function; it must be added separately as a layer).
            One of :func:`relu`, :func:`tanh`, :func:`sigmoid`, or :class:`PReLULayer`
        :param bool use_bias: `True` for bias, `False` for no bias.  No bias should
            be used when batch normalization layer
            will be processing the output
            of this layer (e.g. when :meth:`add_batch_normalization` is called next).
        """
        pool_stride = pool_shape if pool_stride is None else pool_stride
        weight_init = self.activation if weight_init is None else weight_init

        preceding = self.layers[-1]

        assert len(preceding.out_shape) == 4

        layer = _ConvPoolLayer(
            inp=preceding.output,
            inp_shape=preceding.out_shape,
            filter_shape=(num_filters, preceding.out_shape[1],
                          filter_shape[0], filter_shape[1]),
            pool_size=pool_shape,
            pool_stride=pool_stride,
            weight_init=weight_init,
            use_bias=use_bias,
            rng=self.rng
        )
        self.layers.append(layer)

    def add_convolution(self, num_filters, filter_shape, pool_shape,
                        pool_stride=None, activation=None, batch_normalize=True):
        """
        Adds a convolution, pooling layer and nonlinearity to the network
        (with the option of a batch normalization layer).

        :param int num_filters: number of filter maps to create
        :param tuple filter_shape: two dimensional shape of filters
        :param tuple pool_shape: two dimensional shape of pools
        :param tuple pool_stride: distance between pool starting points; if this is
            less than `pool_shape` then pools will be overlapping
        :param activation: activation function to be applied to pool output.
            (One of :func:`relu`, :func:`tanh`, :func:`sigmoid`, or :class:`PReLULayer`)
        :param bool batch_normalize: `True` for batch normalization, `False` for no
            batch normalization.
        """

        activation = self.activation if activation is None else activation
        pool_stride = pool_shape if pool_stride is None else pool_stride
        self.add_conv_pool(
            num_filters=num_filters,
            filter_shape=filter_shape,
            weight_init=activation,
            pool_shape=pool_shape,
            pool_stride=pool_stride,
            use_bias=not batch_normalize)
        if batch_normalize:
            self.add_batch_normalization()
        self.add_nonlinearity(activation)

    def add_fully_connected(self, num_units, weight_init, use_bias):
        """
        Add a layer that does matrix multiply and addition of biases.
        (No nonlinearity is applied
        in this layer because when batch normalization is applied it must
        come between the matrix multiply and the nonlinearity.  A nonlinearity
        can be applied either by using the :meth:`add_hidden` convenience
        method instead of this one or by subsequently calling :meth:`add_nonlinearity`.)

        :param int num_units: number of neurons in the fully connected layer
        :param weight_init: activation function that will be applied
            after the :class:`_FullyConnectedLayer` (used
            to determine a weight initialization scheme--one of :func:`relu`,
            :func:`tanh`, :func:`sigmoid`, or :class:`PReLULayer`)
        :param bool use_bias: ``True`` to use bias; ``False`` not to.  (When
            using batch normalization, bias is redundant and thus should not be used.)
        """
        weight_init = self.activation if weight_init is None else weight_init
        inp, inp_shape = _two_dimensional(self.layers[-1])
        layer = _FullyConnectedLayer(
            rng=self.rng,
            inp=inp,
            n_in=inp_shape[1],
            n_out=num_units,
            batch_size=self.batch_size,
            weight_init=weight_init,
            use_bias=use_bias)
        self.layers.append(layer)

    def add_hidden(self, num_units, activation=None, batch_normalize=True):
        """
        Add a hidden layer consisting of a fully connected layer, a nonlinearity
        layer, and optionally a batch normalization layer.  (The equivalent of
        calling :meth:`add_fully_connected`, :meth:`add_batch_normalization`,
        and :meth:`add_nonlinearity` in sequence.)

        :param int num_units: number of neurons in the hidden layer
        :param activation: activation function to be applied
        :param bool batch_normalize: `True` for batch normalization, `False` for no
            batch normalization.
        """
        activation = self.activation if activation is None else activation
        self.add_fully_connected(num_units, activation, not batch_normalize)
        if batch_normalize:
            self.add_batch_normalization()
        self.add_nonlinearity(activation)

    def add_nonlinearity(self, nonlinearity):
        """
        Add a layer which applies a nonlinearity to its inputs.

        :param nonlinearity: the activation function to be applied.
            (One of :func:`relu`, :func:`tanh`, :func:`sigmoid`, or :class:`PReLULayer`)
        """
        preceding = self.layers[-1]

        if nonlinearity in (relu, tanh, sigmoid):
            layer = _NonLinearityLayer(
                inp=preceding.output,
                inp_shape=preceding.out_shape,
                nonlinearity=nonlinearity)
        elif nonlinearity == PReLULayer:
            layer = PReLULayer(
                inp=preceding.output,
                inp_shape=preceding.out_shape)
        else:
            raise Exception('unsupported nonlinearity:' + nonlinearity)

        self.layers.append(layer)

    def add_dropout(self, rate=0.5):
        """
        Add a dropout layer.

        :param float rate: rate at which to randomly zero out inputs
        """
        layer = _DropoutLayer(
            inp=self.layers[-1].output,
            inp_shape=self.layers[-1].out_shape,
            rate=rate,
            train_flag=self.train_flag,
            srng=self.srng
        )
        self.layers.append(layer)

    def add_batch_normalization(self):
        """
        Add a batch normalization layer
        """
        layer = _BatchNormalizationLayer(
            inp=self.layers[-1].output,
            inp_shape=self.layers[-1].out_shape,
            train_flag=self.train_flag,
            rng=self.rng)
        self.layers.append(layer)

    def add_logistic(self):
        """
        Add a logistic classifier (should be the final layer).
        """
        inp, inp_shape = _two_dimensional(self.layers[-1])
        layer = _LogisticRegression(
            inp=inp,
            y=self.y,
            n_in=inp_shape[1],
            n_out=self.num_classes)
        self.layers.append(layer)

    def add_mlp(self, num_hidden_units, activation=None):
        """
        A convenience function for adding a hidden layer and logistic regression
        layer at the same time.  (Mostly here to mirror deeplearning.net tutorial.

        :param int num_hidden_units: number of hidden units
        :param activation: activation function to be applied to hidden layer output.
            (One of :func:`relu`, :func:`tanh`, :func:`sigmoid`, or :class:`PReLULayer`)
        """
        activation = self.activation if activation is None else activation

        self.add_hidden(num_hidden_units, activation)
        self.add_logistic()

    def _get_params_grad(self):
        return reduce(operator.add, [layer.params for layer in self.layers
                                     if hasattr(layer, 'params')])

    def _get_params_other(self):
        tuples = reduce(operator.add, [layer.updates for layer in self.layers
                                       if hasattr(layer, 'updates')], [])
        return zip(*tuples)[0] if len(tuples) > 0 else []

    def _get_params(self):
        return ([pm.get_value(borrow=False) for pm in self._get_params_grad()],
                [pm.get_value(borrow=False) for pm in self._get_params_other()])

    def _set_params(self, params):
        for param, best in zip(self._get_params_grad(), params[0]):
            param.set_value(best, borrow=False)
        for param, best in zip(self._get_params_other(), params[1]):
            param.set_value(best, borrow=False)

    def _make_givens(self, training, data_x, data_y=None):
        """
        makes the ``givens`` substitution list passed as an argument to
        ``theano.function``.

        :param bool training: `True` if training, `False` if not training.  A flag is
            created for :class:`_DropoutLayer` and :class:`_BatchNormalizationLayer`
            to use when building symbolic graphs since they compute different
            transformations when training and not training.
        :param shared data_x: the training, validation or test data being used by
            the function
        :param shared data_y: the training, validation or test classes being used by
            the function
        :return: list of symbolic substitutions to be passed as the `givens`
            argument
        """
        begin = self.index * self.batch_size
        end = (self.index + 1) * self.batch_size
        givens = {}
        for name, symbol in self.x_symbols.items():
            givens[symbol] = data_x[name][begin: end]

        if data_y is not None:
            givens[self.y] = data_y[begin: end]
        else:
            assert training is False

        if any([layer.__class__ in (_DropoutLayer, _BatchNormalizationLayer)
                for layer in self.layers]):
            if training:
                givens[self.train_flag] = np.int8(1)
            else:
                givens[self.train_flag] = np.int8(0)
                # if training:
                # givens[self.train_flag_switch] = T.as_tensor_variable(
                #         1.0, 'train_flag_on')
                # else:
                #     givens[self.train_flag_switch] = T.as_tensor_variable(
                #         -1.0, 'train_flag_off')
        return givens

    def train(self, updater, epochs=200, final_epochs=0, l1_reg=0, l2_reg=0):
        """
        Train the model

        :param updater: the learning rule; one of :class:`StochasticGradientDescent`,
            :class:`Adam`, :class:`AdaDelta`, or :class:`Momentum`
        :param int epochs: the number of epochs to train for
        :param int final_epochs: the number of final epochs to train for.  (Final
            epochs are epochs where the validation and test data are folded
            into the training data for a little boost in the size of the dataset.)
        :param float l1_reg: l1 regularization penalty
        :param float l2_reg: l2 regularization penalty
        """
        print('building model...')
        start_time = time.time()
        batch_size = self.batch_size

        train_x, train_y, train_y_uncast = \
            _shared_dataset(*self.preprocessor.get_train())
        valid_x, valid_y, _ = \
            _shared_dataset(*self.preprocessor.get_valid())
        test_x, test_y, _ = \
            _shared_dataset(*self.preprocessor.get_test())

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_x.values()[0].get_value(
            borrow=True).shape[0] // batch_size
        n_valid_batches = valid_x.values()[0].get_value(
            borrow=True).shape[0] // batch_size
        n_test_batches = test_x.values()[0].get_value(
            borrow=True).shape[0] // batch_size

        last_layer = self.layers[-1]

        # create a list of all model parameters to be fit by gradient descent
        params = self._get_params_grad()

        l1 = reduce(operator.add, [abs(param).sum() for param in params])
        l2 = reduce(operator.add, [(param ** 2).sum() for param in params])

        # the cost we minimize during training is the NLL of the
        # model plus the L1 and L2 regularization
        cost = last_layer.negative_log_likelihood + l1_reg * l1 + l2_reg * l2
        multi_class_loss = last_layer.negative_log_likelihood
        errors = last_layer.errors

        print('compiling validation and test functions...')

        test_model = theano.function(
            [self.index],
            (multi_class_loss, errors),
            givens=self._make_givens(False, test_x, test_y)
        )

        validate_model = theano.function(
            [self.index],
            (multi_class_loss, errors),
            givens=self._make_givens(False, valid_x, valid_y)
        )

        print('compiling training function...')

        updates = updater._get_updates(params, cost)
        for layer in self.layers:
            if hasattr(layer, 'updates'):
                updates += layer.updates

        output = [multi_class_loss, errors]

        # TODO: figure out how to debug print in theano without getting exceptions
        # some values are printed, but I get a lot of exceptions in the process...

        # global theano_print
        # print("####### theano_print ##########: ", theano_print)
        # output += theano_print

        # def detect_nan(i, node, fn):
        # for output in fn.outputs:
        #         if (not isinstance(np.random.RandomState, output[0]) and
        #                 np.isnan(output[0]).any()):
        #             print '*** NaN detected ***'
        #             theano.printing.debugprint(node)
        #             print 'Inputs : %s' % [input[0] for input in fn.inputs]
        #             print 'Outputs: %s' % [output[0] for output in fn.outputs]
        #             break

        # TODO: do nans when monitoring mode turned on indicate bugs?
        # when either of the debug/monitoring commented out below are
        # on I get exceptions and nans even for code and models that has
        # seemed to work for a long time... what gives?
        train_model = theano.function(
            [self.index],
            output,
            updates=updates,
            # mode='DebugMode',
            # mode=theano.compile.MonitorMode(
            #             post_func=detect_nan).excluding(
            #     'local_elemwise_fusion', 'inplace'),
            givens=self._make_givens(True, train_x, train_y)
        )

        print('finished building model ({})'.format(time.time() - start_time))

        print('training with {}...'.format(updater.__class__.__name__))

        def run_batches(func, num_batches):
            losses, errs = zip(*[func(i) for i in range(num_batches)])
            return np.mean(losses), np.mean(errs)

        best_valid_loss = np.inf
        best_valid_error = np.inf
        best_train_loss = np.inf

        header = ('-' * 100 + '\n epoch | valid err, loss | train err, loss   | ' +
                  'test err, loss  | other \n' + '-' * 101)
        print(header)
        train_start_time = time.time()
        try:
            for epoch in range(epochs):
                train_string = ''
                other_string = ''
                start_time = time.time()
                train_loss, train_error = run_batches(train_model, n_train_batches)
                valid_loss, valid_error = run_batches(validate_model, n_valid_batches)

                train_string += ' {:0>3d}   | {:3.1%}, {:7.6f} | '.format(
                    epoch, valid_error, valid_loss)
                train_string += '{:3.1%}, {:7.6f} {} | '.format(
                    train_error, train_loss,
                    '*' if train_loss > best_train_loss else ' ')
                if train_loss < best_train_loss:
                    best_train_loss = train_loss

                if epoch == 0:
                    other_string += 'train time: {:.1f}, '.format(
                        time.time() - start_time)

                if valid_loss < best_valid_loss:
                    start_time = time.time()
                    save(self, self.output_dir + os.sep + self.name + '_best_net.pkl')
                    best_valid_loss = valid_loss
                    best_valid_error = valid_error
                    test_loss, test_error = run_batches(test_model, n_test_batches)
                    train_string += '{:3.1%}, {:7.6f} | '.format(test_error, test_loss)
                    if epoch == 0:
                        other_string += 'test time: {:.1f}, '.format(
                            time.time() - start_time)
                else:
                    train_string += '                | '

                stop, message = updater._epoch_results(self, train_loss, valid_loss)
                other_string += message
                print(train_string + other_string)

                if stop:
                    break

                new_train_x, new_train_y = self.preprocessor.get_train()
                for key in new_train_x:
                    train_x[key].set_value(new_train_x[key])
                train_y_uncast.set_value(new_train_y)
        except KeyboardInterrupt:
            print('ending principle training')
        finally:
            print(('best validation error {:3.1%}, {:7.6f}; '
                   'trained for {:.1f} minutes').format(
                best_valid_error, best_valid_loss,
                (time.time() - train_start_time) / 60))

        try:
            if final_epochs != 0:
                message = updater._final_epochs()
                print(('training on all data (including valid and test sets) ' +
                       'for {} epochs; ').format(final_epochs) + message)

                print('-' * 50 + '\n epoch | train err, loss   | other | \n' +
                      '-' * 50)

                for epoch in range(final_epochs):
                    new_train_x, new_train_y = self.preprocessor.get_all_train()
                    for key in new_train_x:
                        train_x[key].set_value(new_train_x[key])
                    train_y_uncast.set_value(new_train_y)
                    n_final_batches = train_x.values()[0].get_value(borrow=True).shape[
                                          0] // batch_size

                    train_loss, train_error = run_batches(train_model, n_final_batches)
                    print(' {:0>3d}   | {:3.1%}, {:7.6f} | '.format(
                        epoch, train_error, train_loss))
        except KeyboardInterrupt:
            print('ending final training')
        finally:
            save(self, self.output_dir + os.sep + self.name + '_final_net.pkl')

    # decorators mess up sphinx documentation!  :-(
    # @tools.time_once
    def predict(self, data):
        """
        Predict classes for input data.

        :param ndarray data: data to be processed in order to make prediction
        :return: (list of predicted class indexes for each inference observation,
            list of assessed probabilities for each class possibility for each
            inference observation)
        """
        data = self.preprocessor.preprocess(data)
        batch_size = self.batch_size

        # if predictions don't fit evenly into batch size have to pad
        # the batch size for theano
        def add_padding(data, amount):
            new_shape = list(data.shape)
            new_shape[0] += amount
            return np.resize(data, tuple(new_shape))

        num_padded = batch_size - data.values()[0].shape[0] % batch_size
        for key, value in data.items():
            data[key] = add_padding(value, num_padded)

        num_batches = data.values()[0].shape[0] // batch_size

        data = _shared_dataset(data, borrow=True)

        last_layer = self.layers[-1]
        predict_func = theano.function(
            [self.index],
            (last_layer.y_pred, last_layer.p_y_given_x),
            givens=self._make_givens(False, data)
        )

        outs = []

        for i in range(num_batches):
            outs.append(predict_func(i))

        preds = np.concatenate([out[0] for out in outs])
        probs = np.concatenate([out[1] for out in outs])

        # if predictions didn't fit evenly into batch size have to remove
        # the padding
        preds = preds[:-num_padded]
        probs = probs[:-num_padded]

        return preds, probs

    def make_confusion_matrix(self, data, classes, files):
        """
        Make a confusion matrix given input data and correct class designations

        :param ndarray data: the data for which classes are predicted
        :param ndarray classes: the correct classes to be compared with the predictions
        :param files: an id/index for each observation to facilitate connecting
            them back up to filenames
        :return: (confusion matrix, list of mistakes (file_index, actual, pred))
        """
        pred, prob = self.predict(data)

        matrix = np.zeros((self.num_classes, self.num_classes), dtype='uint64')
        mistakes = []
        for pred, actual, file_index in zip(pred, classes, files):
            matrix[actual, pred] += 1
            if pred != actual:
                mistakes.append((file_index, actual, pred))
        return matrix, mistakes

    def get_confusion_matrices(self):
        """
        Run :meth:`make_confusion_matrix` on training, validation and test data
        and return list of results.

        :return: list of confusion matrices for training, validation, and test data
        """
        return [self.make_confusion_matrix(x, y, f) for x, y, f in
                zip(self.preprocessor.get_raw_x(), self.preprocessor.get_raw_y(),
                    self.preprocessor.get_files())]


def _shared_dataset(x, y=None, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason to store datasets in shared variables is to allow
    Theano to copy them into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch every time
    it is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.

    :param ndarray x: data
    :param ndarray y: classes
    :param bool borrow: ``True`` to use ``x`` and ``y`` without copying; ``False``
        to make copies.
    """
    assert all([data.dtype == theano.config.floatX for data in x.values()])
    wrapped_x = {key: theano.shared(item, borrow=borrow) for key, item in x.items()}

    if y is None:
        return wrapped_x
    else:
        assert y.dtype == theano.config.floatX
        wrapped_y = theano.shared(y, borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets us get around this issue
        return wrapped_x, T.cast(wrapped_y, 'int32'), wrapped_y


def save(nn, filename=None):
    """
    save a :class:`NN` object to file

    :param nn: the :class:`NN` to be saved
    :param string filename: the path/filename to save to
    :return: the filename
    """
    with open(filename, mode='wb') as f:
        cPickle.dump(nn, f, protocol=cPickle.HIGHEST_PROTOCOL)
    return filename


def load(filename):
    """
    load a :class:`NN` object from file

    :param string filename: the path/filename to load from
    :return: the :class:`NN` object loaded
    """
    with open(filename, mode='rb') as f:
        return cPickle.load(f)


# def _param_hash(params):
# return np.mean([np.mean(param) for param in params])
#
#
# def _param_count(params):
#     return np.sum([param.size for param in params])


def net_size(root, layers):
    """
    A simple utility to calculate the computational size of the network
    and give a very rough estimate of how long it will take to train.
    (Ignoring the cost of the activation
    function, batch_normalization, prelu parameters, and a zillion other things.)

    :param tuple root: image shape (channels, height, width)
    :param tuple layers: list of layers where each layer is
        either a conv layer specification or a fully connected layer
        specification.  E.g.:
        ('conv', {'filter': (192, 3, 3), 'pool': (3, 3), 'pool_stride': (2, 2)}),
        or ('full', {'num': 3072})

    ..  adds and mults looks like they might take roughly the same time:
        http://stackoverflow.com/questions/1146455/whats-the-relative-speed-of-floating-point-add-vs-floating-point-multiply
    """
    num_calcs = 0
    num_params = 0
    num_neurons = 0
    prev_out = root

    print('\ntype | calcs | params |    neurons     | '
          '    output     | mults | adds | weights | biases | pool loss\n' + '-' * 102)

    for kind, params in layers:
        if kind == 'conv':
            inp = prev_out
            conv_out = (params['filter'][0],
                        (inp[1] - params['filter'][1] + 1),
                        (inp[2] - params['filter'][2] + 1))
            pool_out = (conv_out[0],
                        int(ceil((conv_out[1] - params['pool'][0] + 1)
                                 / params['pool_stride'][0])),
                        int(ceil((conv_out[2] - params['pool'][1] + 1)
                                 / params['pool_stride'][1])))

            pool_loss = ((conv_out[1] - params['pool'][0]) % params['pool_stride'][0],
                         (conv_out[2] - params['pool'][1]) % params['pool_stride'][1])
            pool_loss = '' if pool_loss[0] == 0 and pool_loss[1] == 0 else pool_loss

            mults = (
                # for each filter map in the layer
                conv_out[0] *
                # for every pixel in the filter map
                conv_out[1] * conv_out[2] *
                # for every filter map in the previous layer
                inp[0] *
                # there is a weight multiply for each of
                # the (rows * cols) pixels in the filter
                params['filter'][1] * params['filter'][2]
            )

            adds = (
                # each of the products above must be summed
                # with some combination of the others
                mults +
                # and a bias added for each output pixel (maps * rows * cols)
                conv_out[0] * conv_out[1] * conv_out[2]
            )

            weights = (
                # there is a weight for each pixel in each filter
                params['filter'][0] * params['filter'][1] * params['filter'][2]
            )

            # one bias for each filter
            biases = params['filter'][0]

            print(
            '{} |{:>6} |{:>7} |{:>15} |{:>15} |{:>6} |{:>5} |{:>8} |{:>7} |{:>10}'
            .format(kind, tools.h(mults + adds), tools.h(weights + biases),
                    conv_out, pool_out, tools.h(mults), tools.h(adds),
                    tools.h(weights), tools.h(biases), pool_loss))

            num_calcs += mults + adds
            num_params += weights + biases
            num_neurons += conv_out[0] * conv_out[1] * conv_out[2]
            prev_out = pool_out
        elif kind == 'full':
            if len(prev_out) == 1:
                prev = prev_out[0]
            elif len(prev_out) == 3:
                prev = prev_out[0] * prev_out[1] * prev_out[2]
            else:
                raise Exception('blah 0')
            mults = prev * params['num']
            adds = mults + params['num']

            weights = mults
            biases = params['num']

            print('{} |{:>6} |{:>7} |{:>15} |{:>15} |{:>6} |{:>5} |{:>8} |{:>7} |'
                  .format(kind, tools.h(mults + adds), tools.h(weights + biases),
                          params['num'], params['num'], tools.h(mults), tools.h(adds),
                          tools.h(weights), tools.h(biases)))

            num_calcs += mults + adds
            num_params += weights + biases
            num_neurons += params['num']
            prev_out = [params['num']]
        else:
            raise Exception('blah 1')
    print('-' * 40 + '\ntotal {:>6}{:>9}{:>17}'.format(
        tools.h(num_calcs), tools.h(num_params), num_neurons))
    print('\n{:.2f} mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2'.
          format(num_calcs * 1e-6 / 60))
    print('{:.1f} mb estimated for model'.format(num_params * 8 / 2 ** 20))


"""
n.add_convolution(72, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
n.add_dropout(p.dropout_conv)
n.add_convolution(128, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
n.add_dropout(p.dropout_conv)
n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
n.add_dropout(p.dropout_conv)
n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
n.add_dropout(p.dropout_conv)
n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
n.append_data_channel('shapes')
n.add_batch_normalization()
n.add_nonlinearity(p.activation)
n.add_dropout(p.dropout_conv)
n.add_hidden(3072, batch_normalize=p.batch_normalize)
n.add_dropout(p.dropout_hidd)
n.add_hidden(3072, batch_normalize=p.batch_normalize)
n.add_dropout(p.dropout_hidd)
n.add_hidden(3072, batch_normalize=p.batch_normalize)
n.add_dropout(p.dropout_hidd)
n.add_logistic()
"""

if __name__ == '__main__':
    print('\ne076 (actual ?? mins/epoch):')
    net_size(
        (1, 77, 77),
        (
            ('conv', {'filter': (96, 7, 7), 'pool': (3, 3), 'pool_stride': (2, 2)}),
            ('conv', {'filter': (128, 5, 5), 'pool': (3, 3), 'pool_stride': (2, 2)}),
            ('conv', {'filter': (128, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
            ('conv', {'filter': (192, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
            ('conv', {'filter': (192, 3, 3), 'pool': (3, 3), 'pool_stride': (2, 2)}),
            ('full', {'num': 3072}),
            ('full', {'num': 3072}),
            ('full', {'num': 3072})
        )
    )

    # print('\ne069 (actual ?? mins/epoch):')
    # net_size(
    #     (1, 77, 77),
    #     (
    #         ('conv', {'filter': (64, 7, 7), 'pool': (3, 3), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (96, 5, 5), 'pool': (3, 3), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (96, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (128, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (128, 3, 3), 'pool': (3, 3), 'pool_stride': (2, 2)}),
    #         ('full', {'num': 3072}),
    #         ('full', {'num': 3072})
    #     )
    # )

    # print('\ne067 (actual ?? mins/epoch):')
    # net_size(
    #     (1, 78, 78),
    #     (
    #         ('conv', {'filter': (72, 7, 7), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (128, 5, 5), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (128, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (192, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (128, 3, 3), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('full', {'num': 3072}),
    #         ('full', {'num': 3072}),
    #         ('full', {'num': 3072})
    #     )
    # )

    # print('\n64 (actual 4.08 mins/epoch):')
    # net_size(
    #     (1, 70, 70),
    #     (
    #         ('conv', {'filter': (64, 7, 7), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (96, 5, 5), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (96, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (128, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (128, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('full', {'num': 3072}),
    #         ('full', {'num': 3072})
    #     )
    # )

    # print('\ne063 (actual 3.9 mins/epoch):')
    # net_size(
    #     (1, 70, 70),
    #     (
    #         ('conv', {'filter': (64, 7, 7), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (96, 5, 5), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (96, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (128, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (128, 3, 3), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('full', {'num': 3072}),
    #         ('full', {'num': 3072})
    #     )
    # )

    # print('\ne057 (actual 3.3 mins/epoch):')
    # net_size(
    #     (1, 64, 64),
    #     (
    #         ('conv', {'filter': (64, 7, 7), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (96, 5, 5), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (96, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (128, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (128, 3, 3), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('full', {'num': 2048}),
    #         ('full', {'num': 2048})
    #     )
    # )
    #
    # print('\ne056 (actual 5.2 mins/epoch):')
    # net_size(
    #     (1, 64, 64),
    #     (
    #         ('conv', {'filter': (32, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (64, 3, 3), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (96, 5, 5), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (128, 5, 5), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('full', {'num': 1280}),
    #         ('full', {'num': 1280})
    #     )
    # )
    #
    # print('\ne058 (actual 2.5 mins/epoch):')
    # net_size(
    #     (1, 64, 64),
    #     (
    #         ('conv', {'filter': (32, 7, 7), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (64, 5, 5), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (96, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (128, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (128, 3, 3), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('full', {'num': 1280}),
    #         ('full', {'num': 1280})
    #     )
    # )
    #
    # print('\ne059 (actual 5.4 mins/epoch):')
    # net_size(
    #     (1, 64, 64),
    #     (
    #         ('conv', {'filter': (32, 5, 5), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (64, 5, 5), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (64, 3, 3), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('conv', {'filter': (96, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
    #         ('conv', {'filter': (64, 3, 3), 'pool': (2, 2), 'pool_stride': (2, 2)}),
    #         ('full', {'num': 1280}),
    #         ('full', {'num': 1280})
    #     )
    # )
    pass