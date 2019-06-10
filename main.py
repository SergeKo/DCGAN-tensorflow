import os
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables, inverse_transform
from time import gmtime, strftime


flags = tf.app.flags
flags.DEFINE_integer("epoch", 50, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 1000000000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("data_dir", "./../data", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("data_list", None, "The file list (if used, input_fname_pattern is ignored). Pathes should be relative to data_dir")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("test", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("generate", False, "True for generating set of images, False for training [False]")
flags.DEFINE_integer("generate_num", 10, "Number of images to generate [10]")
flags.DEFINE_float("generate_min_D_prob", None, "Min. discriminator output score for generated images [None]")
flags.DEFINE_string("generate_name_postfix", "" , "Postfix of generated images")
flags.DEFINE_string("generate_dir", "./../generated" , "Path to generated images")
flags.DEFINE_integer("test_mode", 7, "Visualizing mode: 0 - ; 1 - ; 2 - ;3 - ;4 - ;5 - ;6 - ;7 -")
flags.DEFINE_integer("z_dim", 100, "Latent space size")
flags.DEFINE_integer("y_dim", None, " space size")
flags.DEFINE_boolean("use_double_G_train", True, "use_double_G_train")
flags.DEFINE_boolean("add_dense", False, "Adding dense layer on top of generator [False]")
flags.DEFINE_boolean("generate_known_z", False, "Generate using predefined z values")
flags.DEFINE_string("known_z_path", "./../data", "File with mean Z values to use for generation")

FLAGS = flags.FLAGS

def main(_):
  pp(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=FLAGS.y_dim,
          z_dim=FLAGS.z_dim,
          dataset_name=FLAGS.dataset,
          data_dir=FLAGS.data_dir,
          data_list=FLAGS.data_list,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          use_double_G_train=FLAGS.use_double_G_train,
          add_deconv=FLAGS.add_dense)

    show_all_variables()

    if FLAGS.train:
        train_dcgan(dcgan)

    if not FLAGS.train:
        if not dcgan.load(FLAGS.checkpoint_dir)[0]:
            raise Exception("[!] Train a model first, then run test mode")

    if FLAGS.test:
        test_dcgan(dcgan, sess)

    if FLAGS.generate:
        generated_file_pathes = generate_images(dcgan, sess)

    if FLAGS.generate_known_z:
        z_mean = np.loadtxt(FLAGS.known_z_path)
        generated_file_pathes = generate_images(dcgan, sess, z_mean)


def train_dcgan(dcgan):
    dcgan.train(FLAGS)


def test_dcgan(dcgan, sess):
    visualize(sess, dcgan, FLAGS, FLAGS.visualize_mode)


def generate_images(dcgan, sess, predefined_mean_z=None, z_sigma=0.05):
    print(FLAGS.generate_name_postfix)
    strLog = ""
    generation_data = []
    dir = FLAGS.generate_dir
    if not os.path.exists(dir):
        os.makedirs(dir)

    imgInd = 0
    while (imgInd < FLAGS.generate_num):
        if predefined_mean_z is None:
            z_sample = np.random.uniform(-1, 1, size=(FLAGS.batch_size, FLAGS.z_dim))
        else:
            z_sample = [[np.random.normal(mu, z_sigma) for mu in z_sigma] for _ in range(FLAGS.batch_size)]
            z_sample = np.array(z_sample)

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

        ps = sess.run(dcgan.D, feed_dict={dcgan.inputs: samples})
        probs = ps[:, 0]

        images = inverse_transform(samples)
        for im, z, d_prob in zip(images, z_sample, probs):
            if imgInd >= FLAGS.generate_num:
                break

            if FLAGS.generate_min_D_prob is None or d_prob >= FLAGS.generate_min_D_prob:
                filename = 'gen_%d_dPr_%.2f.png' % (imgInd, d_prob)
                scipy.misc.imsave(os.path.join(dir, filename), np.squeeze(im))
                strLog += filename + ',' + FLAGS.generate_name_postfix + ',' + strftime("%Y%m%d%H%M%S", gmtime()) + '\n'
                generation_data.append((filename, str(d_prob),','.join([str(_) for _ in z])))
                imgInd += 1

            if imgInd % 50 == 0:
                print(imgInd)

    f = open(dir + "/generated_list.txt", "a")
    f.write(strLog)
    f.close()
    with open(dir + "/generated_info.csv", 'w') as ff:
        ff.write('file,D_prob,' + ','.join(['z' + str(_) for _ in range(0, FLAGS.z_dim)]) + '\n')
        for g_d in generation_data:
            ff.write(','.join(g_d) + '\n')
    # plt.hist(probs)
    # plt.show()
    return [os.path.join(dir, gd[0]) for gd in generation_data]


if __name__ == '__main__':
  tf.app.run()
