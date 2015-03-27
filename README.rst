******************************************************************************
pydnn: High performance GPU neural network library for deep learning in Python
******************************************************************************

pydnn is a deep neural network library written in Python using `Theano <http://deeplearning.net/software/theano/>`_ (symbolic math and optimizing compiler package).  It was written for `Kaggle's National Data Science Bowl <http://www.datasciencebowl.com/>`_ competition in March 2015, where it produced an entry finishing in the `top 6% <http://www.kaggle.com/c/datasciencebowl/leaderboard/private>`_.  Continued development is planned, including support for even more of the most important deep learning techniques (RNNs...)

.. contents::

============
Design Goals
============

* **Simplicity**
    Wherever possible simplify code to make it a clear expression of underlying deep learning algorithms.  Minimize cognitive overhead, so that it is easy for someone who has completed the `deeplearning.net tutorials <http://deeplearning.net/tutorial/>`_ to pickup this library as a next step and easily start learning about, using, and coding more advanced techniques.

* **Completeness**
    Include all the important and popular techniques for effective deep learning and **not** techniques with more marginal or ambiguous benefit.

* **Ease of use**
    Make preparing a dataset, building a model and training a deep network only a few lines of code; enable users to work with NumPy rather than Theano.

* **Performance**
    Should be roughly on par with other Theano neural net libraries so that pydnn is a viable choice for computationally intensive deep learning.

========
Features
========

* High performance GPU training (courtesy of Theano)
* Quick start tools to instantly get started training on `inexpensive <http://aws.amazon.com/ec2/pricing/>`_ Amazon EC2 GPU instances.
* Implementations of important new techniques recently reported in the literature:
    * `Batch Normalization <http://arxiv.org/pdf/1502.03167v3.pdf>`_
    * `Parametric ReLU <http://arxiv.org/pdf/1502.01852.pdf>`_ activation function,
    * `Adam <http://arxiv.org/pdf/1412.6980v4.pdf>`_ optimization
    * `AdaDelta <http://arxiv.org/pdf/1212.5701v1.pdf>`_ optimization
    * etc.
* Implementations of standard deep learning techniques:
    * Stochastic Gradient Descent with Momentum
    * Dropout
    * convolutions with max-pooling using overlapping windows
    * ReLU/Tanh/sigmoid activation functions
    * etc.

=============
Documentation
=============

http://pydnn.readthedocs.org/en/latest/index.html

============
Installation
============

pip install pydnn

=====
Usage
=====

First download and unzip raw image data from somewhere (e.g. Kaggle). Then::

    import pydnn
    import numpy as np
    rng = np.random.RandomState(e.rng_seed)

    # build data, split into training/validation sets, preprocess
    train_dir = 'home\ubuntu\train'
    data = pydnn.data.DirectoryLabeledImageSet(train_dir).build()
    data = pydnn.preprocess.split_training_data(data, 64, 80, 15, 5)
    resizer = pydnn.preprocess.StretchResizer()
    pre = pydnn.preprocess.Rotator360(data, (64, 64), resizer, rng)

    # build the neural network
    net = pydnn.nn.NN(pre, 'images', 121, 64, rng, pydnn.nn.relu)
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

    # train the network
    lr = pydnn.nn.Adam(learning_rate=pydnn.nn.LearningRateDecay(
                learning_rate=0.006,
                decay=.1))
    net.train(lr)

From raw data to trained network (including specifying
network architecture) in 25 lines of code.


================
Short Term Goals
================

* Implement popular RNN techniques.
* Integrate with Amazon EC2 clustering software (such as `StarCluster <http://star.mit.edu/cluster/>`_).
* Integrate with hyper-parameter optimization frameworks (such as `Spearmint <https://github.com/JasperSnoek/spearmint>`_ and `hyperopt <https://github.com/hyperopt/hyperopt>`_).

=======
Authors
=======

Isaac Kriegman