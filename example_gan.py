#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains an example of a GAN generated from the MNIST database, following the
tutorial here_.

.. _here: https://goo.gl/iubWGF
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Callable

from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential, Model
from keras.layers import LeakyReLU, Conv2D
from keras.layers import Dropout, Flatten
from keras.layers import Dense, Activation
from keras.layers import BatchNormalization, UpSampling2D
from keras.layers import Reshape, Conv2DTranspose
from keras.optimizers import RMSprop


class ElapsedTimer(object):
    """
    Starts and runs the elapsed time
    """
    def __init__(
            self,
            current_time_getter: Callable[[], datetime] = datetime.utcnow
    ) -> None:
        self.start_time = current_time_getter()
        self._time_getter = current_time_getter

    def start(self) -> None:
        self.start_time = self._time_getter()

    @property
    def elapsed_time(self) -> timedelta:
        return self._time_getter() - self.start_time

    def formatted_elapsed_time(self, printer=print) -> None:
        printer(
            'Elapsed time: %s' % str(self.elapsed_time)
        )


class DeepConvolutionalGenerativeAdversarialNetwork(object):
    """
    A generative adversarial network for writing handwritten digits
    that look like the MNIST database
    """
    def __init__(
            self,
            image_rows: int = 28,
            image_columns: int = 28,
            channel: int = 1
    ) -> None:
        """

        :param image_rows: The number of rows in the picture
        :param image_columns: The number of columns in the picture
        :param channel: The number of colour channels in the photo
        """
        self.img_rows = image_rows
        self.img_cols = image_columns
        self.channel = channel

    def make_discriminator(self) -> Model:
        discriminator = Sequential()

        depth = 64
        dropout = 0.4

        input_shape = (self.img_rows, self.img_cols, self.channel)

        # Conv2D is a 2D convolutional neural network. strides=2 sets the
        # amount of pixels by which each iteration
        discriminator.add(
            Conv2D(
                depth, 5, strides=2, input_shape=input_shape,
                padding='same'
            )
        )

        # LeakyReLU is an activation function for a layer. It "allows for a
        # small gradient when the unit is not active". f(x) = alpha * x
        discriminator.add(LeakyReLU(alpha=0.2))

        # The dropout layer drops some examples to prevent overfitting
        discriminator.add(Dropout(dropout))

        discriminator.add(
            Conv2D(
                depth * 2, 5, strides=2, input_shape=input_shape,
                padding='same'
            )
        )
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(dropout))

        discriminator.add(
            Conv2D(
                depth * 4, 5, strides=2, input_shape=input_shape,
                padding='same'
            )
        )
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(dropout))

        discriminator.add(
            Conv2D(
                depth * 8, 5, strides=2, input_shape=input_shape,
                padding='same'
            )
        )
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(dropout))

        discriminator.add(Flatten())
        discriminator.add(Dense(1))
        discriminator.add(Activation('sigmoid'))

        return discriminator

    @staticmethod
    def make_generator() -> Model:
        dropout = 0.4
        depth = 64 + 64 + 64 + 64
        dimension = 7

        generator = Sequential()

        generator.add(Dense(dimension * dimension * depth, input_dim=100))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(Activation('relu'))
        generator.add(Reshape((dimension, dimension, depth)))
        generator.add(Dropout(dropout))

        generator.add(UpSampling2D())
        generator.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(Activation('relu'))

        generator.add(UpSampling2D())
        generator.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(Activation('relu'))

        generator.add(UpSampling2D())
        generator.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(Activation('relu'))

        generator.add(Conv2DTranspose(1, 5, padding='same'))
        generator.add(Activation('sigmoid'))

        return generator

    def discriminator_model(self) -> Model:
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        model = Sequential()
        discriminator = self.make_discriminator()

        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    def adversarial_model(self) -> Model:
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        model = Sequential()
        model.add(self.make_generator())
        model.add(self.make_discriminator())
        model.compile(
            loss='binary_crossentropy', optimizer=optimizer,
            metrics=['accuracy']
        )
        return model


class MNISTGenerativeAdversarialNetwork(object):
    """
    Describes The GAN on the MNIST database
    """
    def __init__(self) -> None:
        self.image_rows = 28
        self.image_columns = 28
        self.channel = 1

        self.x_trainer = input_data.read_data_sets(
            "mnist", one_hot=True
        ).train.images

        self.x_train = self.x_train.reshape(
            -1, self.image_rows, self.image_columns, 1).astype(
            np.float32
        )

        self.gan = DeepConvolutionalGenerativeAdversarialNetwork()


    @property
    def discriminator(self) -> Model:
        return self.gan.discriminator_model()