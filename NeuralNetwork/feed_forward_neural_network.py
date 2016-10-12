import sys
import tensorflow as tf
from NeuralNetwork.naural_network_base import NeuralNetworkBase
from run_config_settings import *
from math import floor


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

class FeedForwardNN(NeuralNetworkBase):
    def __init__(self, nr_of_hidden_layers=1, hidden_layers_size=[10], activation_functions=[0, 0], bias=True):
        self.bias = bias
        self.nr_of_hidden_layers = nr_of_hidden_layers
        self.hidden_layers_size = hidden_layers_size
        self.activation_functions = activation_functions


    def construct_neural_network(self, input_size=1000, output_size=7):

        self.input_tensor = tf.placeholder(tf.float32, [None, input_size])

        W_1 = tf.Variable(tf.zeros([input_size, output_size]))
        b = tf.Variable(tf.zeros([output_size]))

        self.activation_function = tf.nn.softmax(tf.matmul(self.input_tensor, W_1) + b)

        self.output_tensor = tf.placeholder(tf.float32, [None, output_size]) # output layer size

        self.init = tf.initialize_all_variables()


    def train_neural_network(self, samples, labels, samples_test, labels_test):
        self.sess = tf.Session()
        self.sess.run(self.init)

        for i in range(TRAINING_ITERATIONS):
            sys.stdout.write("\rTraining network %d%%" % floor((i + 1) * (100/TRAINING_ITERATIONS)))
            sys.stdout.flush()

            cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.output_tensor * tf.log(self.activation_function), reduction_indices=[1]))
            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

            batch_xs, batch_ys = self.get_next_batch(i*BATCH_SIZE, BATCH_SIZE, samples, labels)
            self.sess.run(self.train_step, feed_dict={self.input_tensor: batch_xs, self.output_tensor: batch_ys})
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
        correct_prediction = tf.equal(tf.argmax(self.activation_function, 1), tf.argmax(self.output_tensor, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(self.sess.run(accuracy, feed_dict={self.input_tensor: samples_test, self.output_tensor: labels_test}))
