import tensorflow as tf
import glob
import os


def normalize_image(image):

    image = tf.cast(image, tf.float32) / 255.
    image = (image - 0.5) / 0.5
    return image


def unnormalize_image(image):

    image = (image * 0.5 + 0.5) * 255.
    image = tf.cast(image, tf.uint8)
    return image


def input_data(sess):

    FLAGS = tf.app.flags.FLAGS

    list_images = glob.glob(os.path.join(FLAGS.dataset, "*.jpg"))

    # Read each JPEG file
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(list_images)
    key, value = reader.read(filename_queue)
    channels = FLAGS.channels
    image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")
    image.set_shape([None, None, channels])

    # Crop and other random augmentations
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_saturation(image, .95, 1.05)
    # image = tf.image.random_brightness(image, .05)
    # image = tf.image.random_contrast(image, .95, 1.05)

    # Center crop
    image = tf.image.central_crop(image, FLAGS.central_fraction)

    # Resize
    image = tf.image.resize_images(image, (FLAGS.img_size, FLAGS.img_size), method=tf.image.ResizeMethod.AREA)

    # Normalize
    image = normalize_image(image)

    # Format image to correct ordering
    if FLAGS.data_format == "NCHW":
        image = tf.transpose(image, (2,0,1))

    # Using asynchronous queues
    img_batch = tf.train.batch([image],
                               batch_size=FLAGS.batch_size,
                               num_threads=FLAGS.num_threads,
                               capacity=FLAGS.capacity_factor * FLAGS.batch_size,
                               name='batch_input')

    return img_batch
