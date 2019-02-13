#!/usr/bin/python3
import tensorflow as tf
import argparse
import numpy as np
import os

from dataset.dataset_generator import DatasetGenerator


def cast_image_to_float(image, bbox=[]):

    return tf.cast(image, tf.float32)


def cast_pntltf(altitude_image,
                latitude_image,
                longitude_image,
                class_label,
                score):

    altitude_image = tf.cast(altitude_image, tf.float32)
    latitude_image = tf.cast(latitude_image, tf.float32)
    longitude_image = tf.cast(longitude_image, tf.float32)

    return altitude_image, latitude_image, longitude_image, class_label, score


tf.logging.set_verbosity(tf.logging.ERROR)


def xavier_initialized_matrix(shape):

    input_size = shape[0]
    xavier_stddev = 1. / tf.sqrt(input_size / 2.)
    return(tf.random_normal(shape, stddev=xavier_stddev))


def conv_block(input_feature_map,
               filter_height,
               filter_width,
               in_channels,
               out_channels,
               stride,
               name,
               padding='SAME',
               batch_normalization=False):

    with tf.variable_scope(name):

        # First, construct the filter varible shape...
        filter_size = [filter_height, filter_width, in_channels, out_channels]

        # ...then create a variable of that shape.
        conv_filter = tf.get_variable("filter",
                                      initializer=xavier_initialized_matrix(filter_size))

        # Format stride size, assuming equal strides and unit channel stride.
        stride_size = [1, stride, stride, 1]

        # Build a conv2d operation to operate on the created variables.
        conv_output = tf.nn.conv2d(input=input_feature_map,
                                   filter=conv_filter,
                                   strides=stride_size,
                                   padding=padding)

        # Construct a bia variable with one bias per output channel.
        bias = tf.get_variable("bias", initializer=tf.zeros([out_channels]))

        # Add the bias to the convolutional output to make the local field.
        local_field = conv_output + bias

        # Apply batch normalization.
        if batch_normalization:

            raise NotImplementedError("You haven't implemented batch norm yet")
            tf.nn.batch_normalization()

        # Apply a relu nonlinearity to construct the final feature map.
        output_feature_map = tf.nn.relu(local_field)

        return(output_feature_map)


def compute_output_size(input_size, filter_size, stride, padding):

    # return(((input_size - filter_size + (2 * padding)) / stride) + 1)

    if padding == 'SAME':

        return(np.ceil(float(input_size) / float(stride)))

    if padding == 'VALID':

        return(np.ceil(float(input_size - filter_size + 1) / float(stride)))


def model_fn(input_pipieline):

    (altitude_image,
     latitude_image,
     longitude_image,
     class_label,
     score) = input_pipieline

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):

        input_size = 28

        num_classes = 2

        # TODO: Alternatively, build a FE for each input then concat those.
        x = tf.concat(values=[altitude_image,
                              latitude_image,
                              longitude_image],
                      axis=3)

        with tf.variable_scope("feature_extractor"):

            x = conv_block(input_feature_map=x,
                           filter_height=3,
                           filter_width=3,
                           in_channels=3,
                           out_channels=8,
                           stride=2,
                           name="conv_block_1",
                           padding='SAME',
                           batch_normalization=False)

            output_size = compute_output_size(input_size,
                                              filter_size=3,
                                              padding='SAME',
                                              stride=2)

            print(output_size)

            x = conv_block(input_feature_map=x,
                           filter_height=3,
                           filter_width=3,
                           in_channels=8,
                           out_channels=16,
                           stride=1,
                           name="conv_block_2",
                           padding='SAME',
                           batch_normalization=False)

            output_size = compute_output_size(output_size,
                                              filter_size=3,
                                              padding='SAME',
                                              stride=1)

            print(output_size)

            # conv_size = int(output_size * output_size * 16)

            feature_map = x

        # with tf.variable_scope("flattener"):

        #     conv2_flat = tf.reshape(x,
        #                             shape=[-1, conv_size],
        #                             name="flatten_ofmap_reshape")

        with tf.variable_scope("classification_head"):

            x = conv_block(input_feature_map=feature_map,
                           filter_height=3,
                           filter_width=3,
                           in_channels=16,
                           out_channels=16,
                           stride=1,
                           name="conv_block_1",
                           padding='SAME',
                           batch_normalization=False)

            x = conv_block(input_feature_map=x,
                           filter_height=3,
                           filter_width=3,
                           in_channels=16,
                           out_channels=16,
                           stride=1,
                           name="conv_block_2",
                           padding='SAME',
                           batch_normalization=False)

            x = conv_block(input_feature_map=x,
                           filter_height=3,
                           filter_width=3,
                           in_channels=16,
                           out_channels=16,
                           stride=1,
                           name="conv_block_3",
                           padding='SAME',
                           batch_normalization=False)

            x = conv_block(input_feature_map=x,
                           filter_height=3,
                           filter_width=3,
                           in_channels=16,
                           out_channels=num_classes,
                           stride=1,
                           name="conv_block_4",
                           padding='SAME',
                           batch_normalization=False)

            class_logit = tf.reduce_mean(x, axis=[1, 2])

            for class_id in range(num_classes):

                heatmap = tf.expand_dims(x[:, :, :, class_id], -1)

                print(heatmap.shape)

                tf.summary.image('class_' + str(class_id) + '_heatmap',
                                 heatmap[:, :, :, :1],
                                 max_outputs=4)

            # reg_score = tf.reduce_mean(x)

            # D_W1 = tf.get_variable(name="D_W1",
            #                        initializer=xavier_initialized_matrix([conv_size,
            #                                                               hidden_layer_size]))

            # D_b1 = tf.get_variable(name="D_b1",
            #                        initializer=tf.zeros([hidden_layer_size]))

            # h = tf.nn.relu(tf.matmul(conv2_flat, D_W1) + D_b1)

            # D_W2 = tf.get_variable(name="D_W2",
            #                        initializer=xavier_initialized_matrix([hidden_layer_size,
            #                                                               num_classes]))

            # # D_b2 = tf.get_variable(name="D_b2",
            # #                        initializer=tf.zeros([hidden_layer_size, num_classes]))

            # class_logit = tf.matmul(h, D_W2)
            # # logit = logit + D_b2

            # prob = tf.nn.sigmoid(logit)

        with tf.variable_scope("score_head"):

            x = conv_block(input_feature_map=feature_map,
                           filter_height=3,
                           filter_width=3,
                           in_channels=16,
                           out_channels=16,
                           stride=1,
                           name="conv_block_1",
                           padding='SAME',
                           batch_normalization=False)

            x = conv_block(input_feature_map=x,
                           filter_height=3,
                           filter_width=3,
                           in_channels=16,
                           out_channels=16,
                           stride=1,
                           name="conv_block_2",
                           padding='SAME',
                           batch_normalization=False)

            x = conv_block(input_feature_map=x,
                           filter_height=3,
                           filter_width=3,
                           in_channels=16,
                           out_channels=16,
                           stride=1,
                           name="conv_block_3",
                           padding='SAME',
                           batch_normalization=False)

            x = conv_block(input_feature_map=x,
                           filter_height=3,
                           filter_width=3,
                           in_channels=16,
                           out_channels=1,
                           stride=1,
                           name="conv_block_4",
                           padding='SAME',
                           batch_normalization=False)

            # x = tf.nn.sigmoid(x)

            tf.summary.image('score_heatmap',
                             x[:, :, :, :3],
                             max_outputs=4)

            reg_score = tf.reduce_mean(x)

            # reg_head_W1 = tf.get_variable(name="reg_head_W1",
            #                               initializer=xavier_initialized_matrix([conv_size,
            #                                                                      hidden_layer_size]))

            # reg_head_b1 = tf.get_variable(name="reg_head_b1",
            #                               initializer=tf.zeros([hidden_layer_size]))

            # h = tf.nn.relu(tf.matmul(conv2_flat, reg_head_W1) + reg_head_b1)

            # reg_head_W2 = tf.get_variable(name="reg_head_W2",
            #                               initializer=xavier_initialized_matrix([hidden_layer_size,
            #                                                                      1]))

            # # reg_head_b2 = tf.get_variable(name="reg_head_b2",
            # #                               initializer=tf.zeros([hidden_layer_size, num_classes]))

            # reg_score = tf.matmul(h, reg_head_W2)
            # logit = logit + D_b2

        # prob = tf.nn.sigmoid(logit)

        return(class_logit, reg_score)


def class_loss_fn(model_output_class, target_class):

    with tf.variable_scope("class_loss"):

        model_output_class_print_op = tf.print("\nmodel_output_class: ",
                                               model_output_class)
        target_class_print_op = tf.print("\ntarget_class: ",
                                         target_class)

        # model_output_class, target_class
        with tf.control_dependencies([model_output_class_print_op,
                                      target_class_print_op]):

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model_output_class,
                labels=target_class)

        tf.summary.scalar('sum_class_loss', tf.reduce_sum(loss))

    return(loss)


def reg_loss_fn(model_output_score, target_score):

    with tf.variable_scope("score_loss"):

        model_output_score_print_op = tf.print("\nmodel_output_score: ",
                                               model_output_score)

        target_score_print_op = tf.print("\ntarget_score: ",
                                         target_score)

        # model_output_class, target_class
        with tf.control_dependencies([model_output_score_print_op,
                                      target_score_print_op]):

            loss = tf.losses.mean_squared_error(predictions=[model_output_score],
                                                labels=target_score)

        tf.summary.scalar('sum_reg_loss', tf.reduce_sum(loss))

    return(loss)


def ptnltf_model(data_pipeline, model, class_loss_fn, reg_loss_fn):

    (altitude_image,
     latitude_image,
     longitude_image,
     class_label,
     score) = data_pipeline

    with tf.variable_scope("image_summaries"):

        # Add summaries to the graph for instrumentation.
        tf.summary.image('altitude_image',
                         altitude_image[:, :, :, :3],
                         max_outputs=4)
        tf.summary.image('latitude_image',
                         latitude_image[:, :, :, :3],
                         max_outputs=4)
        tf.summary.image('longitude_image',
                         longitude_image[:, :, :, :3],
                         max_outputs=4)

    model_output_class, model_output_score = model(data_pipeline)

    with tf.variable_scope("loss"):

        class_loss = class_loss_fn(model_output_class, class_label)

        class_loss_print_op = tf.print("\nclass_loss: ", class_loss)

        score_loss = reg_loss_fn(model_output_score, score)

        score_loss_print_op = tf.print("\nscore_loss: ", score_loss)

        with tf.control_dependencies([class_loss_print_op,
                                      score_loss_print_op]):

            loss = class_loss + (10 * score_loss)

    with tf.variable_scope("optimizer"):

        optimizer_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return(optimizer_op)


def train(FLAGS):

    train_tfrecord_name = os.path.join(FLAGS.dataset_path,
                                       FLAGS.train_tfrecord)

    with tf.variable_scope("data_pipeline"):

        # Instantiate a wrapper for the dataset generation process.
        # Potential problem: this encoding fn is casting all data to floats...
        train_generator = DatasetGenerator(train_tfrecord_name,
                                           num_channels=FLAGS.num_channels,
                                           augment=FLAGS.augment_train_data,
                                           shuffle=FLAGS.shuffle_train_data,
                                           batch_size=FLAGS.batch_size,
                                           num_threads=FLAGS.num_dataset_threads,
                                           buffer=FLAGS.dataset_buffer_size,
                                           encoding_function=cast_pntltf)

        train_iterator = train_generator.dataset.make_initializable_iterator()

        data_pipeline = train_iterator.get_next()

    train_op = ptnltf_model(data_pipeline,
                            model_fn,
                            class_loss_fn,
                            reg_loss_fn)

    merge_op = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(
        os.path.join("./tensorboard/", '{}_train'.format(FLAGS.name)),
        graph=tf.get_default_graph())

    # Instantiate a saver.
    saver = tf.train.Saver()

    # train_writer = tf.summary.FileWriter(
    #     os.path.join("./tensorboard/",
    #                  '{}_train_{}'.format("minimal_gan",
    #                                       time.strftime('%d-%m_%I_%M'))),
    #     graph=tf.get_default_graph())

    # Create a session.
    with tf.Session() as sess:

        # Initialize global and local variables.
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        global_step = 0

        # Run the main training loop.
        for epoch in range(FLAGS.num_epochs):

            print("\n=============\nEpoch: " + str(epoch) + "\n============\n")

            sess.run(train_iterator.initializer)

            epoch_step = 0

            # Until the training iterator is exhausted...
            while True:

                print("\nEpoch step: " + str(epoch_step) + "\n")

                try:

                    print("training.")

                    _, summary = sess.run([train_op, merge_op])

                    train_writer.add_summary(summary, global_step)

                    global_step += 1

                    epoch_step += 1

                except tf.errors.OutOfRangeError:

                    print('End of epoch: ' + str(epoch))

                    if epoch_step % 128 == 0:

                        summary = sess.run(merge_op)

                        train_writer.add_summary(summary, global_step)


                        ckpt_path = os.path.join(FLAGS.checkpoint_dir,
                                                 '{}_last.ckpt'.format(FLAGS.name))

                        save_path = saver.save(sess,
                                               ckpt_path,
                                               global_step=epoch)

                        print('Model saved to: ', save_path)

                    break


def main(**kwargs):

    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    parser.add_argument('--name', type=str,
                        default="ipnetv0",
                        help='Name of this model.')

    parser.add_argument('--dataset_path', type=str,
                        default="C:\\research\\ptn-ltf\\data\\tfrecords\\",
                        help='Path to the training TFRecord file.')

    parser.add_argument('--train_tfrecord', type=str,
                        default="ptnltf_0.tfrecords",
                        help='Name of the training TFRecord file.')

    parser.add_argument('--valid_tfrecord', type=str,
                        default="ptnltf_1.tfrecords",
                        help='Name of the validation TFRecord file.')

    parser.add_argument('--test_tfrecord', type=str,
                        default="ptnltf_0.tfrecords",
                        help='Name of the testing TFRecord file.')

    parser.add_argument('--checkpoint_dir', type=str,
                        default="C:\\research\\ptn-ltf\\checkpoints\\",
                        help='Path to the training TFRecord file.')

    parser.add_argument('--dataset_buffer_size', type=int,
                        default=512,
                        help='Number of images to prefetch in the input pipeline.')

    parser.add_argument('--num_epochs', type=int,
                        default=512,
                        help='Number of training epochs to run')

    parser.add_argument('--num_channels', type=int,
                        default=1,
                        help='Number of channels in the input data.')

    parser.add_argument('--num_dataset_threads', type=int,
                        default=64,
                        help='Number of threads to be used by the input pipeline.')

    parser.add_argument('--batch_size', type=int,
                        default=1,
                        help='Batch size to use in training, validation, and testing/inference.')

    parser.add_argument('--augment_train_data', type=bool,
                        default=False,
                        help='If True, augment the training data')

    parser.add_argument('--shuffle_train_data', type=bool,
                        default=True,
                        help='If True, shuffle the training data')

    FLAGS = parser.parse_args()

    # Launch training
    train(FLAGS)


if __name__ == '__main__':

    main()
