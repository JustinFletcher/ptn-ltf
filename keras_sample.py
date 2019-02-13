
import sys
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras


class SampleModel(keras.Model):

    def __init__(self, num_classes=10):

        super(SampleModel, self).__init__(name='my_model')
        self.num_classes = num_classes

        # Define your layers here.
        self.dense_1 = keras.layers.Dense(32, activation='relu')
        self.dense_2 = keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        """
        Define your forward pass here, using layers you previously defined in
        `__init__`).
        """

        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want
        # to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


def generate_toy_dataset(num_samples=1):

    # Make toy data.
    data = np.random.random((num_samples, 32))
    labels = np.random.random((num_samples, 10))

    # Instantiates a toy dataset instance.
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    dataset = dataset.repeat()
    return(dataset)

def generate_toy_image(num_samples=1):

    # Make toy data.
    data = np.random.random((num_samples, 32))
    labels = np.random.random((num_samples, 10))

    # Instantiates a toy dataset instance.
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    dataset = dataset.repeat()
    return(dataset)


def main(_):

    dataset = generate_toy_dataset(num_samples=1000)
    val_dataset = generate_toy_dataset(num_samples=100)

    if FLAGS.train_with_keras_fit:

        # Instantiates the subclassed model.
        sample_model = SampleModel(num_classes=10)

        # The compile step specifies the training configuration.
        sample_model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

        sample_model.fit(dataset, epochs=100, steps_per_epoch=30,
                         validation_data=val_dataset,
                         validation_steps=3)

    if FLAGS.train_with_estimator:

        # Instantiates the subclassed model.
        sample_model = SampleModel(num_classes=10)

        # The compile step specifies the training configuration.
        sample_model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

        # Create an Estimator from the compiled Keras model. Note the initial
        # model state of the keras model is preserved in the created Estimator.
        sample_est = tf.keras.estimator.model_to_estimator(
            keras_model=sample_model)

        sample_est.train(input_fn=generate_toy_dataset, steps=2000)


if __name__ == '__main__':

    # Instantiates an arg parser
    parser = argparse.ArgumentParser()

    # Establishes default arguments
    parser.add_argument("--output_dir",
                        type=str,
                        default="C:\\research\\data\\xview\\tfrecords\\",
                        help="The complete desired output filepath.")

    parser.add_argument("--train_with_estimator",
                        type=bool,
                        default=True,
                        help="")

    parser.add_argument("--train_with_keras_fit",
                        type=bool,
                        default=True,
                        help="")

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Runs the tensorflow app
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
