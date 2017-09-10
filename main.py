import tensorflow as tf
from numpy import *
from numpy import *
import model
from PIL import Image
from resize import *
batch_size=21
x = tf.placeholder(tf.float32, [None,32,24,3])
sess = tf.Session()
with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2,variables = model.convolutional(x)

saver = tf.train.Saver(variables)
saver.restore(sess, "E:/xinlun/data/convolutional.ckpt")
def convolutional1(input):
    return sess.run(y2, feed_dict={x: input}).flatten().tolist()




def mnist():
    #标准化数据
    trainingFileList = listdir("E:/xinlun/batch")
    n = len(trainingFileList)
    # print(trainingFileList)
    train = np.zeros((n, 32, 24, 3))
    # loadrange = random.sample(trainingFileList, batch_size)
    # print(loadrange)
    # output = np.zeros((batch_size, 2))
    output={}
    output2=np.zeros((n,2))
    #遍历图片，并且存放图片和图片名
    list=[]
    j = 0
    for i in trainingFileList:
        fr = Image.open("E:/xinlun/batch/" + i)
        # print(array(fr).shape)
        if array(fr).shape!=([32, 24, 3] or [24,32,3]):
            fr.thumbnail((32, 24),Image.ANTIALIAS)
            # print(fr.shape)
        list.append(i)
        train[j, :] = np.array(fr).reshape([32, 24, 3])
        # print(train.shape)
        j+=1
    output1 = convolutional1(train)

    #将模型输出的256个概率，转化为(batch，2)的矩阵存放
    k = 0
    j = 1
    for a in range(n):
        output2[a,:]=[output1[k],output1[j]]
        # output.append(k)
        k+=2
        j+=2
    #对output的
    for m in range(n):
        if argsort(output2[m,:])[-1]==0:
            output[list[m]]="无车"
        else:
            output[list[m]] = "有车"
    # print(len(output2))
    # print(len(output))
    print(output)
    # print(output2)
    # print(filename+'  convolution result:', argsort(output2)[-1])
    return output
mnist()
# @app.route('/')
# def main():
    # return render_template('index.html')

