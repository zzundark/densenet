import tensorflow as tf
import config as cfg
import numpy as np
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

def Global_average_pooling(x, stride=1):
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) 
	
def Batch_normalization(x, training, scope="bn"):
	with arg_scope([batch_norm],
					scope=scope,
					updates_collections=None,
					decay=0.9,
					center=True,
					scale=True,
					zero_debias_moving_mean=True):
		return tf.cond(training,
						lambda : batch_norm(inputs=x, is_training=training, reuse=None),
						lambda : batch_norm(inputs=x, is_training=training, reuse=True))
def Conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x,num_class) :
    return tf.layers.dense(inputs=x, units=num_class, name='linear')	
	
#모델(model)
class densenet(object):	

	def __init__(self):
		# self.nb_blocks = cfg.nb_blocks
		self.filters = cfg.growth_k
		self.num_class = cfg.num_class
		self.keep_prob=tf.placeholder(tf.float32)
		self.training=tf.placeholder(tf.bool)
		self.images = tf.placeholder(tf.float32, shape=[None,cfg.image_size,cfg.image_size,3], name = 'x_image') 
		self.labels = tf.placeholder(tf.float32, shape=[None, self.num_class], name = 'y_target')
		self.logits = self.build_network(x_image=self.images,num_class=self.num_class)
		

	def bottleneck_layer(self, x, scope):
		with tf.name_scope(scope):
			x = Batch_normalization(x, training=self.training, scope=scope+'_batch1')
			x = Relu(x)
			x = Conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
			x = Drop_out(x, rate=self.keep_prob, training=self.training)

			x = Batch_normalization(x, training=self.training, scope=scope+'_batch2')
			x = Relu(x)
			x = Conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
			x = Drop_out(x, rate=self.keep_prob, training=self.training)
			return x

	def transition_layer(self, x, scope):
		with tf.name_scope(scope):
			x = Batch_normalization(x, training=self.training, scope=scope+'_batch1')
			x = Relu(x)
			x = Conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
			x = Drop_out(x, rate=self.keep_prob, training=self.training)
			x = Average_pooling(x, pool_size=[2,2], stride=2)

			return x

	def dense_block(self, input_x, nb_layers, layer_name):
		with tf.name_scope(layer_name):
			layers_concat = list()
			layers_concat.append(input_x)

			x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleneck_' + str(0))

			layers_concat.append(x)

			for i in range(nb_layers - 1):
				x = Concatenation(layers_concat)
				x = self.bottleneck_layer(x, scope=layer_name + '_bottleneck_' + str(i + 1))
				layers_concat.append(x)

			return x

	def Dense_net(self, x):
		x = Conv_layer(x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
		x = Max_Pooling(x, pool_size=[3,3], stride=2)
		
		x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense1')
		x = self.transition_layer(x, scope='trans_1')

		x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense2')
		x = self.transition_layer(x, scope='trans_2')

		x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense3')
		x = self.transition_layer(x, scope='trans_3')

		x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense4')

		x = Batch_normalization(x, training=self.training, scope='linear_batch_norm')
		x = Relu(x)
		x = Global_average_pooling(x)
		x = flatten(x)
		return x
		
	def build_network(self,x_image,num_class):
		x=self.Dense_net(x_image) 
		x=Linear(x,num_class)
		return x		
		
		
		
