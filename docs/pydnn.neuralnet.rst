pydnn.neuralnet module
======================

.. py:module:: pydnn.neuralnet

Overview
--------

:class:`NN` is the workhorse of pydnn.  Using an instance of :class:`NN` the user defines the network, trains the network and uses the network for inference. :class:`NN` takes care of the bookkeeping and wires the layers together, calculating any intermediate configuration necessary for doing so without user input.  See the :ref:`section on NN <NN_section>` for more details.

Learning rules define how the network updates weights based on the gradients calculated during training.  Learning rules are passed to :class:`NN` objects when calling :func:`NN.train` to train the network.  :class:`Momentum` and :class:`Adam` are good default choices. See :ref:`Learning_Rules` for more details.

All the learning rules defined in this package depend in part on a global learning rate that effects how all parameters are updated on training passes.  It is  frequently beneficial to anneal the learning rate over the course of training and different approaches to annealing can result in substantially different convergence losses and times.  Different approaches to annealing can be achieved by using one of the various learning rate annealing objects which are passed to :class:`LearningRule` objects during instantiation.  :class:`LearningRateDecay` is a good default choice.  See :ref:`Learning_Rates` for more details.

A variety of activation functions, or nonlinearities, can be applied to layers.  :func:`relu` is the most common, however :class:`PReLULayer` has recently been reported to achieve state of the art results.  See :ref:`Activations` for more details.

Finally there are a few utilities for saving and reloading trained networks and for estimating the size and training time for networks before training.  See :ref:`Utilities` for more details.

.. contents::

.. _NN_section:

The main class: NN
------------------

.. autoclass:: NN
   :members:
   :member-order: bysource

.. _Learning_Rules:

Learning Rules (Optimization Methods)
-------------------------------------

.. autoclass:: LearningRule
.. autoclass:: StochasticGradientDescent
   :members:
   :inherited-members:
.. autoclass:: Momentum
   :members:
   :inherited-members:
.. autoclass:: Adam
   :members:
   :inherited-members:
.. autoclass:: AdaDelta
   :members:
   :inherited-members:

.. _Learning_Rates:

Learning Rate Annealing
-----------------------

.. autoclass:: LearningRateAdjuster
.. autoclass:: LearningRateDecay
   :members:
   :inherited-members:
.. autoclass:: LearningRateSchedule
   :members:
   :inherited-members:
.. autoclass:: WackyLearningRateAnnealer
   :members:
   :inherited-members:

.. _Activations:

Activation Functions (Nonlinearities)
-------------------------------------

.. autofunction:: relu
.. autofunction:: prelu
.. autofunction:: sigmoid
.. autofunction:: tanh

.. _Utilities:

Utility Functions
-----------------

.. autofunction:: save
.. autofunction:: load
.. autofunction:: net_size
