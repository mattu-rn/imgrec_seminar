# mnistダウンロード
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# 画像
x = tf.placeholder(tf.float32,shape=[None,784])
x_image = tf.reshape(x, [-1,28,28,1])

# 正解ラベル
y_ = tf.placeholder(tf.float32,shape=[None,10])

# 各種行列を生成する関数
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
  
# convolutionとpoolingを実行する関数
def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
  
# 畳み込み1
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 畳み込み2
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全結合層
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

# 全結合層
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
y = tf.nn.softmax( h_fc2 )


# 目的関数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 最適化手法
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)