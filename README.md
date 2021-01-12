# CNN_MNIST
TensorflowでCNNを書いた時の備忘録

### CNNの備忘録

In [7]:

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
%matplotlib inline
```

### ① Create Tensor ( Variable, Placeholder )

In [40]:

```python
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
```

In [41]:

```python
def weight_initializer( shape ):
    init = tf.truncated_normal( shape, stddev=0.1 ) 
    return tf.Variable(init)

def bias_initializer( shape ):
    init = tf.constant(0.1, shape = shape )
    return tf.Variable(init)
```

In [52]:

```python
w = weight_initializer([4,3])
b = bias_initializer([3,1])
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(w))
    print(sess.run(b))
```

Out [52]:

```
[[ 0.00610536 -0.05607848  0.11475813]
 [-0.08212297  0.05584015 -0.01216668]
 [-0.10446066 -0.01168909 -0.118546  ]
 [ 0.11035935  0.06917038  0.14418903]]
[[0.1]
 [0.1]
 [0.1]]
```

このようにwとbは初期化されることがわかる

### ② Define the model

In [59]:

```python
def conv2d(x, w):
    return tf.nn.conv2d(x,w, strides=[1,1,1,1], padding='SAME')
```

In [69]:

```python
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
```

#### INPUT -> CONV -> RELU -> POOL -> CONV -> RELU -> POOL -> FC1 -> FC2

In [98]:

```python
input_x = tf.reshape(x, [-1,28,28,1])

filter_1 = weight_initializer([5,5,1,32])
b_1 = bias_initializer([32])
CONV1 = tf.nn.relu( conv2d(input_x, filter_1 ) + b_1 )
POOL1 = max_pool_2x2(CONV1)

filter_2 = weight_initializer([5,5,32,64])
b_2 = bias_initializer([64])
CONV2 = tf.nn.relu( conv2d( POOL1, filter_2 ) + b_2 )
POOL2 = max_pool_2x2(CONV2) 

w_3 = weight_initializer([7*7*64, 1024])
b_3 = bias_initializer([1024])
POOL2_flat = tf.reshape(POOL2, [-1, 7*7*64])
FC1 = tf.nn.relu( tf.matmul( POOL2_flat, w_3 ) + b_3 ) 

FC1_drop = tf.nn.dropout(FC1, 0.8)

w_4 = weight_initializer([1024,10])
b_4 = bias_initializer([10])
y_out = tf.matmul( FC1_drop, w_4) + b_4
```

#### 誤差関数の定義

In [92]:

```python
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out))
train = tf.train.AdamOptimizer(1e-4).minimize( cost )
predict = tf.equal( tf.argmax( y, 1 ), tf.argmax( y_out, 1 ) )
accuracy = tf.reduce_mean( tf.cast( predict, tf.float32 ) )
```

### ③ Create the Session

In [96]:

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y:batch[1]})
            print(i, " : ", train_accuracy)
        train.run(feed_dict={x:batch[0], y:batch[1]})
    print('test accuracy : ', accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels}))
```

Out [96]:

```
0  :  0.14
100  :  0.94
200  :  0.86
300  :  0.9
400  :  0.92
500  :  1.0
600  :  0.96
700  :  0.94
800  :  0.92
900  :  1.0
test accuracy :  0.9632
```

