"""
Multilayer Perception.
Author: Hung Tran
Date: 2017-11-24
"""
from __future__ import print_function
#MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import time
import tensorflow as tf
class Config():
	# training params
	lr = 0.01
	n_epochs = 50
	batch_size = 100
	display_step = 1
	# network params
	n_hidden_1 = 256
	n_hidden_2 = 256
	n_input = 784 # image size 28*28
	n_classes = 10 # 0-9 digits

class multiLayerPerceptron():
	def __init__(self,config):
		self.config = config
		self.add_placeholder()
		self.pred = self.add_prediction_op()
		self.loss = self.add_loss_op(self.pred) 
		self.train_op = self.add_train_op(self.loss)
	def add_placeholder(self):
		self.input_placeholder = tf.placeholder(tf.float32, [None,self.config.n_input])
		self.label_placeholder = tf.placeholder(tf.float32,[None,self.config.n_classes])
	def add_prediction_op(self):
		weights = {
		'h1': tf.Variable(tf.random_normal([self.config.n_input,self.config.n_hidden_1])),
		'h2': tf.Variable(tf.random_normal([self.config.n_hidden_1,self.config.n_hidden_2])),
		'out': tf.Variable(tf.random_normal([self.config.n_hidden_2,self.config.n_classes]))
		}
		biases = {
		'b1': tf.Variable(tf.random_normal([self.config.n_hidden_1])),
		'b2': tf.Variable(tf.random_normal([self.config.n_hidden_2])),
		'out': tf.Variable(tf.random_normal([self.config.n_classes]))
		}
		layer_1 = tf.nn.relu(tf.add(tf.matmul(self.input_placeholder,weights['h1']), biases['b1']))
		layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,weights['h2']), biases['b2']))
		pred = tf.matmul(layer_2,weights['out']) + biases['out']
		return pred
	def add_loss_op(self,pred):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_placeholder,logits=self.pred))
		return loss
	def add_train_op(self,loss):
		train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
		return train_op
	def create_feed_dict (self,inputs_batch, labels_batch):
		feed_dict = {self.input_placeholder:inputs_batch,self.label_placeholder:labels_batch}
		return feed_dict
	def train_on_batch(self,sess,inputs_batch, labels_batch):
		feed = self.create_feed_dict(inputs_batch,labels_batch)
		_,loss = sess.run([self.train_op, self.loss], feed_dict=feed)
		return loss
	def run_epoch (self,sess,train_examples):
		total_batch = int(train_examples.train.num_examples/self.config.batch_size)
		total_loss = 0
		for i in range(total_batch):
			inputs_batch, labels_batch = train_examples.train.next_batch(self.config.batch_size)
			total_loss += self.train_on_batch(sess,inputs_batch,labels_batch)
		return total_loss/total_batch

	def fit(self,sess,train_examples):
		losses = []
		for epoch in range(self.config.n_epochs):
			start_time = time.time()
			avg_loss = self.run_epoch(sess,train_examples)
			duration = time.time() - start_time
			print("Epoch {:}:loss = {:.2f} ({:.3f} sec)".format(epoch,avg_loss,duration))
			losses.append(avg_loss)
		return losses
def test_multiLayerPerceptron():
	config = Config()
	mnist = input_data.read_data_sets("./tmp/data", one_hot=True)
	# tf Graph input
	
	with tf.Graph().as_default():
		model = multiLayerPerceptron(config)
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			losses = model.fit(sess,mnist)

			# Test model
			correct_prediction = tf.equal(tf.argmax(model.pred, 1), tf.argmax(model.label_placeholder, 1))
			# Calculate accuracy
			feed = model.create_feed_dict(mnist.test.images,mnist.test.labels)
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			# print("Accuracy:", accuracy.eval({model.input_placeholder: mnist.test.images, model.label_placeholder: mnist.test.labels}))
			print ("Accuracy:", accuracy.eval(feed))

if __name__ == "__main__":
	test_multiLayerPerceptron()





