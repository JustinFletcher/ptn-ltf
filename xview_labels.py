
import io
import os
import sys
import csv
import json
import argparse

import numpy as np

from PIL import Image
from itertools import zip_longest

from matplotlib import pyplot as plt
import tensorflow as tf


class BoundingBox(object):

    def __init__(self, image_filepath, coords, class_id):

        self.image_filepath = image_filepath
        self.coords = coords
        self.class_id = class_id

    def summary(self):

        print("Bounding Box:")
        print("Image Filepath: %s" % self.image_filepath)
        print("Object Class: %d" % self.class_id)
        print(self.coords)


class XViewExample(object):

    """docstring for xViewExample"""

    def __init__(self, image_filepath):

        super(XViewExample, self).__init__()

        self.image_filepath = image_filepath
        self.bounding_boxes = list()

    def add_bounding_box(self, bounding_box):

        self.bounding_boxes.append(bounding_box)

    def summary(self):

        print(self.image_filepath)
        print(self.bounding_boxes)


class ObjectDetectionDataset(object):

    """docstring for ObjectDetectionDataset"""

    def __init__(self):

        super(ObjectDetectionDataset, self).__init__()
        self._examples = {}

    def add_bounding_box_to_example(self, example_id, bounding_box):

        if example_id not in self._examples:

            self._examples[example_id] = XViewExample(example_id)

        self._examples[example_id].add_bounding_box(bounding_box)

    def get_examples_list(self):

        return(list(self._examples.values()))


def get_labels(filename="C:\\research\\data\\xview\\raw\\xView_train.geojson"):

    print("Extracting individual annotations from %s" % filename)

    with open(filename) as f:

        data = json.load(f)

    bounding_boxes = list()

    for feature in data['features']:

        properties = feature['properties']

        bb_coords = properties['bounds_imcoords']

        if bb_coords != []:

            bb = BoundingBox(properties['image_id'],
                             [int(c) for c in bb_coords.split(",")],
                             properties['type_id'])

            bounding_boxes.append(bb)

    return bounding_boxes


def build_dataset_from_annotations(bounding_boxes):

    xview_dataset = ObjectDetectionDataset()

    for i, bounding_box in enumerate(bounding_boxes):

        if i % 1000 == 0:

            print("Processing BoundingBox %d / %d" % (i, len(bounding_boxes)))

        # In this instance, we're using image filepath as ID, so we extract it.
        image_filepath = bounding_box.image_filepath

        # Ensure that there is, in fact, an image for this annotation.
        if os.path.isfile(FLAGS.data_dir + image_filepath):

            xview_dataset.add_bounding_box_to_example(image_filepath,
                                                      bounding_box)

    return(xview_dataset)


def get_xview_examples():

    bounding_boxes = get_labels()

    xview_dataset = build_dataset_from_annotations(bounding_boxes)

    return(xview_dataset.get_examples_list())


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_xview_tf_example(example):

    with io.BytesIO() as f:

        image_filepath = FLAGS.data_dir + example.image_filepath

        try:

            image_array = np.array(Image.open(image_filepath))

        except FileNotFoundError:

            print("There is no image for example %s" % example.image_filepath)

        im = Image.fromarray(image_array)

        im.save(f, format='JPEG')

        # TODO: read and convert to JPEG
        jpeg_encoded_image = f.getvalue()

    # Store the height and width of the image.
    image_height = image_array.shape[0]
    image_width = image_array.shape[1]

    # Declare lists to hold the bounding box values ffor encoding.
    bbox_xmins = list()
    bbox_ymins = list()
    bbox_xmaxs = list()
    bbox_ymaxs = list()
    classes = list()

    # Iterate over each bounding box in this example.
    for bounding_box in example.bounding_boxes:

        # Parse the coords for the bounding box, which are stored postionally.
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bounding_box.coords

        # Normalize these box locations/classes and append them to thier lists.
        bbox_xmins.append(bbox_xmin / image_width)
        bbox_ymins.append(bbox_ymin / image_height)
        bbox_xmaxs.append(bbox_xmax / image_width)
        bbox_ymaxs.append(bbox_ymax / image_height)
        classes.append(bounding_box.class_id)

    # Construct a single feature dict for this example.
    feature_dict = {'image/height': int64_feature(image_height),
                    'image/width': int64_feature(image_width),
                    'image/encoded': bytes_feature(jpeg_encoded_image),
                    'image/format': bytes_feature('jpeg'.encode('utf-8')),
                    'image/object/bbox/xmin': float_list_feature(bbox_xmins),
                    'image/object/bbox/xmax': float_list_feature(bbox_xmaxs),
                    'image/object/bbox/ymin': float_list_feature(bbox_ymins),
                    'image/object/bbox/ymax': float_list_feature(bbox_ymaxs),
                    'image/object/class/label': int64_list_feature(classes)}

    # Encapsulate the features in a TF Features.
    tf_features = tf.train.Features(feature=feature_dict)

    # Build a TF Example.
    tf_example = tf.train.Example(features=tf_features)

    return(tf_example)


def group_list(ungrouped_list, group_size, padding=None):

    grouped_list = zip_longest(*[iter(ungrouped_list)] * group_size,
                               fillvalue=padding)

    return(grouped_list)


def create_xview_tfrecords():

    # Parse the provided GEOJSON into a list of example objects.
    xview_examples = get_xview_examples()

    xview_example_groups = group_list(xview_examples,
                                      FLAGS.examples_per_tfrecord)

    for group_index, xview_example_group in enumerate(xview_example_groups):

        print("Saving group %s" % str(group_index))

        output_path = FLAGS.output_dir + 'xview_' + \
            str(group_index) + '.tfrecords'

        # Open a writer to the provided TFRecords output location.
        with tf.python_io.TFRecordWriter(output_path) as writer:

            # For each example...
            for example in xview_example_group:

                if example:

                    # ...construct a TF Example object...
                    tf_example = create_xview_tf_example(example)

                    # ...and write it to the TFRecord.
                    writer.write(tf_example.SerializeToString())


def view_example_image():

    sample_tfrecord = FLAGS.output_dir + 'xview_5.tfrecords'

    reader = tf.python_io.tf_record_iterator(sample_tfrecord)

    examples = [tf.train.Example().FromString(s) for s in reader]

    example = examples[2]

    image_bytes = example.features.feature['image/encoded'].bytes_list.value[0]

    image = Image.open(io.BytesIO(image_bytes))

    data = np.array(image)

    plt.imshow(data, interpolation='nearest')

    plt.show()

    # image.show()


def parse_geojson_to_csv_labels(
        filename="C:\\research\\data\\xview\\raw\\xView_train.geojson",
        partition=True,
        train_partition_fraction=0.8,
        val_partition_fraction=0.1,
        train_partition_fraction=0.1,
        excluded_image_ids=[]):

    print("Extracting individual annotations from %s" % filename)

    with open(filename) as f:

        data = json.load(f)

    if not partition:

        csv_filename = "C:\\research\\data\\xview\\raw\\xView_labels.csv"

        with open(csv_filename, 'w', newline='') as csvfile:

            csvwriter = csv.writer(csvfile)

            bounding_boxes = list()

            for i, feature in enumerate(data['features']):

                properties = feature['properties']

                bb_coords = properties['bounds_imcoords']

                # Some images in xView are broken; exclude them.
                if not (properties['image_id'] in excluded_image_ids):

                    if bb_coords != []:

                        bb = list()

                        bb.append(properties['image_id'])

                        for c in bb_coords.split(","):

                            bb.append(int(c))

                        bb.append(int(properties['type_id']))

                        bounding_boxes.append(bb)

                    # csvwriter.writerow(bb)
            csvwriter.writerows(bounding_boxes)

    else:


def main(_):

    if FLAGS.make_tfrecords:

        create_xview_tfrecords()

    if FLAGS.view_tfrecords:

        view_example_image()

    if FLAGS.parse_geojson_to_csv_labels:

        excluded_image_ids = ["1395.tif"]

        parse_geojson_to_csv_labels(excluded_image_ids)


if __name__ == '__main__':

    # Instantiates an arg parser
    parser = argparse.ArgumentParser()

    # Establishes default arguments
    parser.add_argument("--output_dir",
                        type=str,
                        default="C:\\research\\data\\xview\\tfrecords\\",
                        help="The complete desired output filepath.")

    parser.add_argument("--examples_per_tfrecord",
                        type=int,
                        default=5,
                        help="The number of examples in a single .tfrecord.")

    parser.add_argument("--data_dir",
                        type=str,
                        default="C:\\research\\data\\xview\\raw\\train_images\\",
                        help="The directory in which raw data is stored.")

    parser.add_argument("--make_tfrecords",
                        type=bool,
                        default=False,
                        help=".")

    parser.add_argument("--view_tfrecords",
                        type=bool,
                        default=True,
                        help=".")

    parser.add_argument("--parse_geojson_to_csv_labels",
                        type=bool,
                        default=True,
                        help=".")

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Runs the tensorflow app
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

