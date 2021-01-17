import tensorflow as tf
import timeit
# 1.创建输入张量,并赋初始值
a = tf.constant(2.)
b = tf.constant(4.)
n = tf.constant(500000)
print('a+b=',a+b)
with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([1, n])
    cpu_b = tf.random.normal([n, 1])
    print(cpu_a.device, cpu_b.device)
# 创建使用 GPU 环境运算的 2 个矩阵
with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([1, n])
    gpu_b = tf.random.normal([n, 1])
    print(gpu_a.device, gpu_b.device)
def cpu_run(): # CPU 运算函数
    with tf.device('/cpu:0'):
         c = tf.matmul(cpu_a, cpu_b)
    return c
def gpu_run():# GPU 运算函数
    c = tf.matmul(cpu_a, cpu_b)

    with tf.device('/gpu:0'):
         c = tf.matmul(gpu_a, gpu_b)
    return c
# 第一次计算需要热身,避免将初始化时间结算在内
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('warmup:', cpu_time, gpu_time)
# 正式计算 10 次,取平均时间
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('run time:', cpu_time, gpu_time)