import os
import time
import tensorflow as tf
from resize import *
import model as model

batch_size=128
max_steps = 3000
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
# model
with tf.variable_scope("convolutional"):
    image_holder = tf.placeholder(tf.float32, [batch_size, 32, 24, 3])
    #解决过拟合问题
    y, variables = model.convolutional(image_holder)
# y1=tf.reshape(y,[128])
# train
label_holder = tf.placeholder(tf.int32, [batch_size])
print('y',y)
print('label_holder',label_holder )

loss = loss(y, label_holder)

# label_holder=tf.to_float(label_holder)

# loss = -tf.reduce_mean(label_holder * tf.log(y))
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)  # 0.72
#AdamOptimizer 数据量大，比梯度下降算法要快些
top_k_op = tf.nn.in_top_k(y, label_holder, 1)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

tf.train.start_queue_runners()
saver = tf.train.Saver(variables)
for step in range(max_steps):
    start_time = time.time()
#     image_batch, label_batch =random.sample(trainingFileList, number)sess.run([images_train, labels_train])
    image_batch, label_batch =loadData(batch_size)
    # print(label_batch.shape)
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch,
                                                          label_holder: label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

num_examples = 772
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_test,label_test = loadTestData(batch_size)
    predictions = sess.run([top_k_op],feed_dict={image_holder: image_test,
                                                 label_holder:label_test})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)

path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'E:/xinlun/data/','convolutional.ckpt'),
        write_meta_graph=False, write_state=False)
print("Saved:", path)