pydnn.preprocess module
=======================

.. py:module:: pydnn.preprocess

Overview
--------

Most of the code in this module is currently pretty specific to processing images like those in Kaggle's plankton competition.  Those images were unique in that they (1) were presented with a uniform background, (2) they varied in size in a way that provided meaningful information about the subject, and (3) they were mostly randomly oriented.  These features have to do with real world constraints on the way that marine biologist collect the images, and are obviously quite different from popular datasets like ImageNet, MNIST, etc.  As I (or others) use pydnn in a greater variety of machine learning contexts a variety of preprocessing approaches can be maintained here.

.. contents::

Training Set
------------

.. autofunction:: split_training_data

.. _Preprocessors:

Preprocessors
-------------

Preprocessors take care of online augmentation, shuffling, zero centering and normalizing, resizing, and other related transformations of the traning data.  Because the plankton images could be in any orientation, to achieve good performance it was important to augment the data with many rotations of the training set so the network could learn to recognize images in different orientations.  Initially I experimented with 90 degree rotations and a flip, however I found that unconstrained degree rotations (:class:`Rotator360` and :class:`Rotator360PlusGeometry`) performed better.  Another approach that I experimented with was rotating and flipping all images into a canonicalized orientation based on their shape and size (:class:`Canonicalizer`), which significantly improves early training progress, but shortly thereafter falls behind a 360 degree rotation approach.

Another thing that these preprocessors do is add additional data channels.  For instance, since, in the case of the plankton dataset, the size of the images carries important information (because image size was related to the size of the organism) it was useful to add a data channel with the original image size (:class:`Rotator360`), because that information is lost when uniformly resizing images to be fed into the neural network.  Another approach, instead of the original image size, was to create a channel with the size of the largest contiguous image shape, and it's rotation in comparison to it's canonicalized rotation (:class:`Rotator360PlusGeometry`).

.. autoclass:: Rotator360
.. autoclass:: Rotator360PlusGeometry
.. autoclass:: Canonicalizer

Resizing
--------

Users do not use the resizers directly but pass them to a :ref:`preprocessor <Preprocessors>` to control how the preprocessor resizes images.

.. autofunction:: Resizer
.. autofunction:: StretchResizer
.. autofunction:: ContiguousBoxPreserveAspectRatioResizer
.. autofunction:: ContiguousBoxStretchResizer
.. autofunction:: ThresholdBoxPreserveAspectRatioResizer
.. autofunction:: ThresholdBoxStretchResizer
.. autofunction:: PreserveAspectRatioResizer
.. autofunction:: StochasticStretchResizer

