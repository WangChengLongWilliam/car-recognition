import tensorflow as tf

# Softmax Regression Model
# Multilayer Convolutional Network
# batch_size=128
def convolutional(image_holder):
    def variable_with_weight_loss(shape, stddev):
        var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
        return var

    weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2
                                        )  # 设置wl为零，是因为第一层神经网路不需要做正则化处理，5x5的卷积核，RGB通道，64个卷积核
    kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')  # 1，1代表每次选取1x1的
    bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    #lrn层
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2)
    kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
    bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
    #lrn层
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    # print('pool2',pool2.shape)
    reshape = tf.reshape(pool2, [-1, 3072])
    # print(reshape.shape)
    # dim = reshape.get_shape()[1].value
    # dim=3072
    weight3 = variable_with_weight_loss(shape=[3072, 384], stddev=0.04)
    bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
    local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

    weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04)
    bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
    local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4) #128x192
    # print(local4)
    weight5 = variable_with_weight_loss(shape=[192,2], stddev=1 / 192.0)
    bias5 = tf.Variable(tf.constant(0.0, shape=[2]))
    logits = tf.add(tf.matmul(local4, weight5), bias5)
    y = logits
    variables = [weight1, bias1, weight2, bias2, weight3, bias3, weight4, bias4,weight5,bias5]
    return y,variables