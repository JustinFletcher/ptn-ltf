
import io
import os
import sys
import argparse

import numpy as np

from PIL import Image
from itertools import zip_longest

from matplotlib import pyplot as plt
import tensorflow as tf


class PTNLTFExample(object):

    """docstring for xViewExample"""

    def __init__(self, image_filepath):

        super(PTNLTFExample, self).__init__()

    def add_class_label(self, class_label):

        self.class_label = class_label

    def add_score(self, score):

        self.score = score

    def add_altitude_image(self, altitude_image):

        self.altitude_image_path = altitude_image

    def add_latitude_image(self, latitude_image):

        self.latitude_image_path = latitude_image

    def add_longitude_image(self, longitude_image):

        self.longitude_image_path = longitude_image

    # TODO: Add other datatype adders here.

    def summary(self):

        print(self.class_label)

        print(self.score)

        print(self.altitude_image_path)

        print(self.latitude_image_path)

        print(self.longitude_image_path)


class PTNLTFDataset(object):

    """docstring for PTNLTFDataset"""

    def __init__(self):

        super(PTNLTFDataset, self).__init__()
        self._examples = {}

    def __maybe_add_example(self, example_id):

        if example_id not in self._examples:

            self._examples[example_id] = PTNLTFExample(example_id)

    def add_class_label_to_example(self, example_id, class_label):

        self.__maybe_add_example(example_id)

        self._examples[example_id].add_class_label(class_label)

    def add_score_to_example(self, example_id, score):

        self.__maybe_add_example(example_id)

        self._examples[example_id].add_score(score)

    def add_altitude_image_to_example(self, example_id, altitude_image):

        self.__maybe_add_example(example_id)

        self._examples[example_id].add_altitude_image(altitude_image)

    def add_latitude_image_to_example(self, example_id, latitude_image):

        self.__maybe_add_example(example_id)

        self._examples[example_id].add_latitude_image(latitude_image)

    def add_longitude_image_to_example(self, example_id, longitude_image):

        self.__maybe_add_example(example_id)

        self._examples[example_id].add_longitude_image(longitude_image)

    # TODO: Add other datatype adders here.

    def get_examples_list(self):

        return(list(self._examples.values()))


def build_dataset_from_folder_structured_raw(data_dir):

    dataset = PTNLTFDataset()

    # Get a list of files in th0e data directory; one file/example.
    files = os.listdir(data_dir)
    print(files)

    # Iterate over each file...
    for example_id, file in enumerate(files):

        filepath = os.path.join(data_dir, file)

        # ...adding the class...
        with open(os.path.join(filepath, 'class.txt'), "r") as f:

            class_label = f.read()
            dataset.add_class_label_to_example(example_id, class_label)

        # ...and the score...
        with open(os.path.join(filepath, 'score.txt'), "r") as f:

            score = f.read()
            dataset.add_score_to_example(example_id, score)

        # ...and the altitude...
        altitude_image = os.path.join(filepath, 'altitude.jpg')
        dataset.add_altitude_image_to_example(example_id, altitude_image)

        # ...and the latitude...
        latitude_image = os.path.join(filepath, 'latitude.jpg')
        dataset.add_latitude_image_to_example(example_id, latitude_image)

        # ...and the longitude...
        longitude_image = os.path.join(filepath, 'longitude.jpg')
        dataset.add_longitude_image_to_example(example_id, longitude_image)

        # TODO: Add support for other inputs.

    return(dataset)


def get_examples():

    dataset = build_dataset_from_folder_structured_raw(FLAGS.data_dir)

    return(dataset.get_examples_list())


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_jpeg_encoded_image(image_filepath):

    with io.BytesIO() as f:

        try:

            image_array = np.array(Image.open(image_filepath))

        except FileNotFoundError:

            print("There is no image at %s" % image_filepath)

        im = Image.fromarray(image_array)

        im.save(f, format='JPEG')

        # TODO: read and convert to JPEG
        jpeg_encoded_image = f.getvalue()

    return(jpeg_encoded_image)


def get_image_height(image_filepath):

    try:

        image_array = np.array(Image.open(image_filepath))

    except FileNotFoundError:

        print("There is no image at %s" % image_filepath)

    return(image_array.shape[0])


def get_image_width(image_filepath):

    try:

        image_array = np.array(Image.open(image_filepath))

    except FileNotFoundError:

        print("There is no image at %s" % image_filepath)

    return(image_array.shape[1])


def create_ptnltf_tf_example(example):

    jpeg_encoded_altitude = get_jpeg_encoded_image(example.altitude_image_path)

    jpeg_encoded_latitude = get_jpeg_encoded_image(example.latitude_image_path)

    jpeg_encoded_longitude = get_jpeg_encoded_image(example.longitude_image_path)

    # Store the height and width of the image.
    image_heights = get_image_width(example.altitude_image_path)
    image_widths = get_image_height(example.altitude_image_path)

    # Construct a single feature dict for this example.
    feature_dict = {'example/class_label': int64_feature(int(example.class_label)),
                    'example/score': float_feature(float(example.score)),
                    'image/height': int64_feature(image_heights),
                    'image/width': int64_feature(image_widths),
                    'image/altitude/encoded': bytes_feature(jpeg_encoded_altitude),
                    'image/latitude/encoded': bytes_feature(jpeg_encoded_latitude),
                    'image/longitude/encoded': bytes_feature(jpeg_encoded_longitude),
                    'image/format': bytes_feature('jpeg'.encode('utf-8'))}

    # TODO: Extend feature dict with more encoded image types.

    # Encapsulate the features in a TF Features.
    tf_features = tf.train.Features(feature=feature_dict)

    # Build a TF Example.
    tf_example = tf.train.Example(features=tf_features)

    return(tf_example)


def group_list(ungrouped_list, group_size, padding=None):

    # Magic, probably.
    grouped_list = zip_longest(*[iter(ungrouped_list)] * group_size,
                               fillvalue=padding)

    return(grouped_list)


def create_tfrecords():

    examples = get_examples()

    example_groups = group_list(examples, FLAGS.examples_per_tfrecord)

    for group_index, example_group in enumerate(example_groups):

        print("Saving group %s" % str(group_index))

        output_path = FLAGS.output_dir + 'ptnltf_' + \
            str(group_index) + '.tfrecords'

        print(output_path)

        # Open a writer to the provided TFRecords output location.
        with tf.python_io.TFRecordWriter(output_path) as writer:

            # For each example...
            for example in example_group:

                if example:

                    # ...construct a TF Example object...
                    tf_example = create_ptnltf_tf_example(example)

                    # ...and write it to the TFRecord.
                    writer.write(tf_example.SerializeToString())


def view_example_image():

    sample_tfrecord = FLAGS.output_dir + 'ptnltf_1.tfrecords'

    reader = tf.python_io.tf_record_iterator(sample_tfrecord)

    examples = [tf.train.Example().FromString(s) for s in reader]

    example = examples[0]

    image_bytes = example.features.feature['image/altitude/encoded'].bytes_list.value[0]

    image = Image.open(io.BytesIO(image_bytes))

    data = np.array(image)

    plt.imshow(data, interpolation='nearest')

    plt.show()

    # image.show()


def main(_):

    if FLAGS.make_tfrecords:

        create_tfrecords()

    if FLAGS.view_tfrecords:

        view_example_image()


if __name__ == '__main__':

    # Instantiates an arg parser
    parser = argparse.ArgumentParser()

    # Establishes default arguments
    parser.add_argument("--output_dir",
                        type=str,
                        default="C:\\research\\ptn-ltf\\data\\tfrecords\\",
                        help="The complete desired output filepath.")

    parser.add_argument("--examples_per_tfrecord",
                        type=int,
                        default=2,
                        help="The number of examples in a single .tfrecord.")

    parser.add_argument("--data_dir",
                        type=str,
                        default="C:\\research\\ptn-ltf\\data\\images",
                        help="The directory in which raw data is stored.")

    parser.add_argument("--make_tfrecords",
                        type=bool,
                        default=True,
                        help=".")

    parser.add_argument("--view_tfrecords",
                        type=bool,
                        default=True,
                        help=".")

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Runs the tensorflow app
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
