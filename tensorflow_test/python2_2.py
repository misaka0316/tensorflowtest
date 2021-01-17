import os
import tensorflow as tf # 导入 TF 库
from tensorflow import keras # 导入 TF 子库 keras
from tensorflow.keras import layers, optimizers, datasets # 导入 TF 子库等
(x, y), (x_val, y_val) = datasets.mnist.load_data() # 加载 MNIST 数据集
x = 2*tf.convert_to_tensor(x, dtype=tf.float32)/255.-1 # 转换为浮点张量,并缩放到
y = tf.convert_to_tensor(y, dtype=tf.int32) # 转换为整形张量
y = tf.one_hot(y, depth=10) # one-hot 编码
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y)) # 构建数据集对象
train_dataset = train_dataset.batch(512)
#In [1]:
y = tf.constant([0,1,2,3]) # 数字编码的 4 个样本标签
y = tf.one_hot(y, depth=10) # one-hot 编码,指定类别总数为 10
print(y)
#Out[1]:
tf.Tensor(
[[1. , 0. , 0.  ,0. , 0.  ,0. , 0. , 0.  ,0.  ,0.] # 数字 0 的 one-hot 编码向量
[0. , 1. , 0. , 0. , 0.,  0.  ,0.  ,0. , 0.  ,0.] # 数字 1 的 one-hot 编码向量
[0.,  0.,  1.,  0.  ,0.  ,0.  ,0., 0. , 0., 0.] # 数字 2 的 one-hot 编码向量
[0.  ,0.  ,0. , 1.,  0.,  0. , 0.,  0.,  0. , 0.]] ,shape=(4, 10), dtype=float32)


# 创建一层网络,设置输出节点数为 256,激活函数类型为 ReLU
layers.Dense(256, activation='relu')
# 利用 Sequential 容器封装 3 个网络层,前网络层的输出默认作为下一层的输入
model = keras.Sequential([ # 3 个非线性层的嵌套模型
    layers.Dense(256, activation='relu'), # 隐藏层 1
    layers.Dense(128, activation='relu'), # 隐藏层 2
    layers.Dense(10)]) # 输出层,输出节点数为 10
with tf.GradientTape() as tape: # 构建梯度记录环境
    # 打平操作,[b, 28, 28] => [b, 784]
    x = tf.reshape(x, (-1, 28*28))
    # Step1. 得到模型输出 output [b, 784] => [b, 10]
    out = model(x)
    # [b] => [b, 10]
    y_onehot = tf.one_hot(y, depth=10)
    # 计算差的平方和,[b, 10]
    loss = tf.square(out-y_onehot)
    # 计算每个样本的平均误差,[b]
    loss = tf.reduce_sum(loss) / x.shape[0]
    # Step3. 计算参数的梯度 w1, w2, w3, b1, b2, b3
    grads = tape.gradient(loss, model.trainable_variables)
    # 自动计算梯度
    grads = tape.gradient(loss, model.trainable_variables)
    # w' = w - lr * grad,更新网络参数
    optimizer.apply_gradients(zip(grads, model.trainable_variables))