import librosa
import tensorflow as tf
import data_loader

data_helper = data_loader.DataLoader()
samples, labels, samples_test, labels_test = data_helper.load_data()

input_size = len(samples[0])
output_size = len(labels[0])

batch_size = 5

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, input_size])
W = tf.Variable(tf.zeros([input_size, output_size]))
b = tf.Variable(tf.zeros([output_size]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, output_size])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train
tf.initialize_all_variables().run()
# for i in range(153):
for i in range(140):
    batch_xs = samples[i*batch_size:(i+1)*batch_size]
    batch_ys = labels[i*batch_size:(i+1)*batch_size]
    train_step.run({x: batch_xs, y_: batch_ys})

print("Done training")
# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: samples_test, y_: labels_test}))
