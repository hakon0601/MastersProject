import sys
import tensorflow as tf
from NeuralNetwork.naural_network_base import NeuralNetworkBase
from run_config_settings import *
from math import floor


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

class FeedForwardNN(NeuralNetworkBase):
    def __init__(self):
        pass


    def construct_neural_network(self, samples, labels, samples_test, labels_test):

        input_size = len(samples[0])
        output_size = len(labels[0])
        # only implemeted one layer of hidden nodes with 100 nodes
        hidden_layer_1_size = 7
        output_layer_size = 7 #TODO fix this as parameter or constant

        self.x = tf.placeholder(tf.float32, [None, input_size])

        W_1 = tf.Variable(tf.zeros([input_size, hidden_layer_1_size]))
        b = tf.Variable(tf.zeros([hidden_layer_1_size]))

        self.y = tf.nn.softmax(tf.matmul(self.x, W_1) + b)

        self.y_ = tf.placeholder(tf.float32, [None, output_size]) # output layer size

        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        #
        # self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        self.init = tf.initialize_all_variables()


    def train_neural_network(self, samples, labels, samples_test, labels_test):
        self.sess = tf.Session()
        self.sess.run(self.init)

        for i in range(TRAINING_ITERATIONS):
            sys.stdout.write("\rTraining network %d%%" % floor((i + 1) * (100/TRAINING_ITERATIONS)))
            sys.stdout.flush()

            cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

            batch_xs, batch_ys = self.get_next_batch(i*BATCH_SIZE, BATCH_SIZE, samples, labels)
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
        print()
        # correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        #
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(self.sess.run(accuracy, feed_dict={self.x: samples_test, self.y_: labels_test}))

    def get_next_batch(self, current_index, batch_size, samples, labels):
        if current_index + batch_size < len(labels):
            return samples[current_index:current_index + batch_size], labels[current_index:current_index + batch_size]
        else:
            end = samples[current_index:], labels[current_index:]
            start = samples[:batch_size - len(end[0])], labels[:batch_size - len(end[1])]
            return end[0] + start[0], end[1] + start[1]

    def test_accuracy_of_solution(self, samples_test, labels_test):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(self.sess.run(accuracy, feed_dict={self.x: samples_test, self.y_: labels_test}))
