import tensorflow as tf
from pretrained.vgg16 import vgg16

class ImageCaptioner(object):
	def __init__(self, config):

		self.config = config

		# Create session
		self.session = tf.Session()

		# Create architecture
		self.cnn_output = None
		self.build_cnn()
		self.build_rnn()

		self.session.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep = 100)


	def build_cnn(self):
		print('Building CNN...')

		if config.cnn_model == 'custom':
			self.build_custom_cnn()

        else:
        	self.build_vgg16()

    def build_custom_cnn(self):
    	print('Building custom model...')

		W_conv1 = _weight_variable([5, 5, 1, 32])
        b_conv1 = _bias_variable([32])
        h_conv1 = tf.nn.relu(_conv2d(imgs_placeholder, W_conv1) + b_conv1)
        h_pool1 = _max_pool_2x2(h_conv1)

        W_conv2 = _weight_variable([5, 5, 32, 64])
        b_conv2 = _bias_variable([64])
        h_conv2 = tf.nn.relu(_conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = _max_pool_2x2(h_conv2)

        W_conv3 = _weight_variable([5, 5, 64, 128])
        b_conv3 = _bias_variable([128])
        h_conv3 = tf.nn.relu(_conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = _max_pool_2x2(h_conv3)
        h_flat3 = tf.reshape(h_pool3, [-1])

        # TODO: Possibly add some FC layers here

        self.cnn_output = h_flat3

    def build_vgg16(self):
    	print('Building VGG-16...')

    	self.imgs_placeholder = tf.placeholder(tf.float32, [None, 224, 224, 3])
    	vgg16(imgs, self.config.cnn_model_file, self.session)
    	self.cnn_output = self.fc2


	def build_rnn(self):
		print('Building RNN...')

		######################
		## BUILD A RNN HERE ##
		######################
		pass		

	def train(self):
		if self.config.train_cnn:
			pass
		else:
			self.sess.run()


	# Layers/initializers
	def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _weight_variable(shape):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)

    def _bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

  	def _max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
