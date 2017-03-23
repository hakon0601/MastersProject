import sys
import tensorflow as tf
from NeuralNetwork.naural_network_base import NeuralNetworkBase
from run_config_settings import *
from math import floor
import random


class ConvolutionalNN(NeuralNetworkBase):
    def __init__(self, hidden_layers=[10, 20], activation_functions_type=[0, 0], cnn_filters=[0, 0], cnn_channels=[0, 0], cnn_strides= [0, 0], pooling_type=0, enable_bias=False, learning_rate=0.5, dropout_rate=0.9, epochs=100, DCL_size=1024):
        self.hidden_layers = hidden_layers
        self.activation_functions_type = activation_functions_type
        self.cnn_filters = cnn_filters
        self.cnn_channels = cnn_channels
        self.cnn_strides = [1 for i in range(len(self.cnn_filters))]
        self.pooling_type = pooling_type
        self.enable_bias = enable_bias
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.DCL_size = DCL_size
        self.original_input_size = 0

    def construct_neural_network(self, input_size=1000):
        self.original_input_size = input_size
        # Create the model
        self.input_tensor = tf.placeholder(tf.float32, [None, input_size[0] * input_size[1]]) # TODO check if this is right, was 512
        self.output_tensor = tf.placeholder(tf.float32, [None, NR_OF_CLASSES])

        self.activation_model = self.model()

        # Train and Evaluate the Model
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.output_tensor * tf.log(self.activation_model), reduction_indices=[1]))
        # self.cost = -tf.reduce_sum(self.output_tensor * tf.log(self.activation_model))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.init = tf.global_variables_initializer()

    def train_neural_network(self, samples, labels, samples_test, labels_test):
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.test_accuracy_of_solution(samples, labels, samples_test, labels_test, reshape=False)

        for epoch in range(self.epochs):
            nr_of_batches_to_cover_all_samples = int(len(samples)/BATCH_SIZE)
            sys.stdout.write("\rTraining CNN network %02d%%\t" % floor((epoch + 1) * (100 / self.epochs)))
            sys.stdout.flush()
            for j in range(nr_of_batches_to_cover_all_samples):
                batch_xs, batch_ys = self.get_random_batch(BATCH_SIZE, samples, labels)
                self.sess.run(self.train_step, feed_dict={self.input_tensor: batch_xs, self.output_tensor: batch_ys, self.keep_prob: self.dropout_rate})

            self.test_accuracy_of_solution(samples, labels, samples_test, labels_test, reshape=False)
        print("Optimization Finished!")

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
        index_of_highest_output_neurons = tf.argmax(self.activation_model, 1)
        index_of_correct_label = tf.argmax(self.output_tensor, 1)
        correct_predictions = tf.equal(index_of_highest_output_neurons, index_of_correct_label)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        accuracy_test = self.sess.run(accuracy, {self.input_tensor: samples_test, self.output_tensor: labels_test, self.keep_prob: 1})
        accuracy_training = self.sess.run(accuracy, {self.input_tensor: samples, self.output_tensor: labels, self.keep_prob: 1})
        print("Accuracy test:", accuracy_test, "Accuracy training:", accuracy_training)

    def model(self):
        x_image = tf.reshape(self.input_tensor, [-1, self.original_input_size[0], self.original_input_size[1], 1]) #

        conv_layers = []

        for i in range(len(self.cnn_filters)):
            if i == 0:
                W_conv1 = tf.Variable(tf.truncated_normal([self.cnn_filters[i], self.cnn_filters[i], 1, self.cnn_channels[i]], stddev=0.1))
                b_conv1 = tf.Variable(tf.constant(0.1 if self.enable_bias else 0.0, shape=[self.cnn_channels[i]]))
                conv = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

            else:
                W_conv2 = tf.Variable(tf.truncated_normal([self.cnn_filters[i], self.cnn_filters[i], self.cnn_channels[i-1], self.cnn_channels[i]], stddev=0.1))
                b_conv2 = tf.Variable(tf.constant(0.1 if self.enable_bias else 0.0, shape=[self.cnn_channels[i]]))
                conv = tf.nn.conv2d(conv_layers[-1], W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2

            if self.activation_functions_type[i] == 0:
                h_conv = tf.nn.tanh(conv)
            elif self.activation_functions_type[i] == 1:
                h_conv = tf.nn.sigmoid(conv)
            elif self.activation_functions_type[i] == 2:
                h_conv = tf.nn.relu(conv)
            elif self.activation_functions_type[i] == 3:
                h_conv = tf.sin(conv)

            if self.pooling_type == 0:
                conv_layers.append(tf.nn.max_pool(h_conv, ksize=[1, POOLING_FILTER_SIZE, POOLING_FILTER_SIZE, 1], strides=[1, POOLING_STRIDE, POOLING_STRIDE, 1], padding='SAME'))
            elif self.pooling_type == 1:
                conv_layers.append(tf.nn.avg_pool(h_conv, ksize=[1, POOLING_FILTER_SIZE, POOLING_FILTER_SIZE, 1], strides=[1, POOLING_STRIDE, POOLING_STRIDE, 1], padding='SAME'))

        cnn_output_w, cnn_output_h = self.calculate_cnn_output_size(original_w=self.original_input_size[0], original_h=self.original_input_size[1])

        # Densely Connected Layer
        W_fc1 = tf.Variable(tf.truncated_normal([int(cnn_output_w*cnn_output_h*self.cnn_channels[-1]), self.DCL_size], stddev=0.1)) #    16*8*32, 64], stddev=0.1))
        # W_fc1 = tf.Variable(tf.truncated_normal([int((32/(2**len(self.hidden_layers)))*(16/(2**len(self.hidden_layers)))*32)], stddev=0.1)) #    16*8*32, 64], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1 if self.enable_bias else 0.0, shape=[self.DCL_size]))

        h_pool2_flat = tf.reshape(conv_layers[-1], [-1, int(cnn_output_w*cnn_output_h*self.cnn_channels[-1])])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Readout Layer
        W_fc2 = tf.Variable(tf.truncated_normal([self.DCL_size, NR_OF_CLASSES], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[NR_OF_CLASSES]))

        return tf.nn.softmax(tf.matmul(self.h_fc1_drop, W_fc2) + b_fc2)

    def calculate_cnn_output_size(self, original_w, original_h):
        w0, h0 = original_w, original_h
        for i in range(len(self.cnn_filters)):
            w = ((w0 - self.cnn_filters[0] + (2*PADDING))/self.cnn_strides[i]) + 1
            h = ((h0 - self.cnn_filters[0] + (2*PADDING))/self.cnn_strides[i]) + 1
            w_after_pool = ((w - POOLING_FILTER_SIZE)/POOLING_STRIDE) + 1
            h_after_pool = ((h - POOLING_FILTER_SIZE)/POOLING_STRIDE) + 1
            w0 = w_after_pool
            h0 = h_after_pool
        return w0, h0
