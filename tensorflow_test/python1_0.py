import os
import tensorflow as tf # 导入 TF 库
from tensorflow import keras # 导入 TF 子库 keras
from tensorflow.keras import layers, optimizers, datasets # 导入 TF 子库等
(x, y), (x_val, y_val) = datasets.mnist.load_data() # 加载 MNIST 数据集
x = 2*tf.convert_to_tensor(x, dtype=tf.float32)/255.-1 # 转换为浮点张量,并缩放到
-1~1
y = tf.convert_to_tensor(y, dtype=tf.int32) # 转换为整形张量
y = tf.one_hot(y, depth=10) # one-hot 编码
print(x.shape, y.shape)
05
train_dataset = tf.data.Dataset.from_tensor_slices((x, y)) # 构建数据集对象
train_dataset = train_dataset.batch(512)
