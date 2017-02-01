import sys
import tensorflow as tf
from NeuralNetwork.naural_network_base import NeuralNetworkBase
from run_config_settings import *
from math import floor
import numpy as np
import random
from tensorflow.python.ops import rnn, rnn_cell

# from tensorflow.examples.tutorials.mnist import input_data


class ConvolutionalNN(NeuralNetworkBase):
    def __init__(self, hidden_layers=[10], activation_functions_type=[0, 0], enable_bias=False, learning_rate=0.5, dropout_rate=0.9, cell_type=0 , time_related_steps=20, epochs=100):
        self.hidden_layers = hidden_layers
        self.activation_functions_type = activation_functions_type
        self.enable_bias = enable_bias
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.time_related_steps = time_related_steps
        self.epochs = epochs

    def construct_neural_network(self, input_size=1000):
        # self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        print("data loaded successfully...")

        self.sess = tf.InteractiveSession()

        # Create the model
        self.x = tf.placeholder(tf.float32, [None, 784])
        x_image = tf.reshape(self.x, [-1,28,28,1])

        self.y_ = tf.placeholder(tf.float32, [None, NR_OF_CLASSES])

        # First Convolutional Layer
        W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second Convolutional Layer
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Densely Connected Layer
        W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) #7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Readout Layer
        W_fc2 = tf.Variable(tf.truncated_normal([1024, NR_OF_CLASSES], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[NR_OF_CLASSES]))

        y_conv=tf.nn.softmax(tf.matmul(self.h_fc1_drop, W_fc2) + b_fc2)


        # Train and Evaluate the Model
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y_conv), reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))





        # tf.reset_default_graph()
        # input_size = int(input_size // self.time_related_steps)
        # self.layers_size = [input_size] + self.hidden_layers + [NR_OF_CLASSES]
        #
        # self.input_tensor = tf.placeholder(tf.float32, [None, self.time_related_steps, self.layers_size[0]])
        # self.output_tensor = tf.placeholder(tf.float32, [None, self.layers_size[-1]])
        # self.keep_prob = tf.placeholder(tf.float32)
        #
        # self.activation_model = self.model()
        # self.cost = -tf.reduce_sum(self.output_tensor * tf.log(self.activation_model))
        # # self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
        # self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        # self.init = tf.global_variables_initializer()

    def train_neural_network(self, samples, labels, samples_test, labels_test):
        # TODO just replace the mnist methods to use the sample produced på the spectrogram class
        self.sess.run(tf.initialize_all_variables())
        for i in range(20000):
            # batch = self.mnist.train.next_batch(50)
            batch_xs, batch_ys = self.get_random_batch(BATCH_SIZE, samples, labels)

            if i%100 == 0:
                train_accuracy = self.accuracy.eval(feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 1.0})
                print("step %d, training accuracy %.7f"%(i, train_accuracy))
            self.train_step.run(feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 0.5})

        crctPreds = np.empty([1, 1])

        for i in range(20000):
            # batch = self.mnist.test.next_batch(50)
            batch_xs, batch_ys = self.get_random_batch(BATCH_SIZE, samples_test, labels_test)

            crctPreds = np.append(crctPreds, np.matrix(self.correct_prediction.eval(feed_dict={self.x:batch_xs, self.y_: batch_ys, self.keep_prob: 1.0})))
            print("Prediction size: ", crctPreds.shape)

        crctPreds.astype(int)
        print()
        print("Accuracy: ",np.mean(crctPreds))



        # tf.get_variable_scope().reuse_variables()
        # samples = self.reshape_samples(samples)
        # samples_test = self.reshape_samples(samples_test)
        # self.sess = tf.Session()
        # self.sess.run(self.init)
        # self.test_accuracy_of_solution(samples, labels, samples_test, labels_test, reshape=False)
        #
        # for epoch in range(self.epochs):
        #     nr_of_batches_to_cover_all_samples = int(len(samples)/BATCH_SIZE)
        #     sys.stdout.write("\rTraining network %02d%%\t" % floor((epoch + 1) * (100 / self.epochs)))
        #     sys.stdout.flush()
        #     for j in range(nr_of_batches_to_cover_all_samples):
        #         batch_xs, batch_ys = self.get_random_batch(BATCH_SIZE, samples, labels)
        #         self.sess.run(self.train_step, feed_dict={self.input_tensor: batch_xs, self.output_tensor: batch_ys, self.keep_prob: self.dropout_rate})
        #
        #     self.test_accuracy_of_solution(samples, labels, samples_test, labels_test, reshape=False)
        # print("Optimization Finished!")

    def get_next_batch(self, current_index, batch_size, samples, labels):
        current_index = current_index % len(samples)
        if current_index + batch_size < len(labels):
            return samples[current_index:current_index + batch_size], labels[current_index:current_index + batch_size]
        else:
            end = samples[current_index:], labels[current_index:]
            start = samples[:batch_size - len(end[0])], labels[:batch_size - len(end[1])]
            return end[0] + start[0], end[1] + start[1]

    def get_random_batch(self, batch_size, samples, labels):
        rand_samples = []
        rand_labels = []
        for i in range(batch_size):
            rand_index = random.randrange(0, len(samples))
            rand_samples.append(samples[rand_index])
            rand_labels.append(labels[rand_index])
        return rand_samples, rand_labels

    def test_accuracy_of_solution(self, samples, labels, samples_test, labels_test, reshape=True):
        # if reshape:
        #     samples = self.reshape_samples(samples)
        #     samples_test = self.reshape_samples(samples_test)

        index_of_highest_output_neurons = tf.argmax(self.activation_model, 1)
        index_of_correct_label = tf.argmax(self.output_tensor, 1)
        correct_predictions = tf.equal(index_of_highest_output_neurons, index_of_correct_label)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        accuracy_test = self.sess.run(accuracy, {self.input_tensor: samples_test, self.output_tensor: labels_test, self.keep_prob: 1})
        accuracy_training = self.sess.run(accuracy, {self.input_tensor: samples, self.output_tensor: labels, self.keep_prob: 1})
        print("Accuracy test:", accuracy_test, "Accuracy training:", accuracy_training)

    def model(self):
        raise NotImplemented

