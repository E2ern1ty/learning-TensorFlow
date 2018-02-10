import tensorflow as tf
import numpy as np
#产生数据
x_data=np.random.rand(100).astype(np.float32)#随机的方式产生x
y_data=x_data*5+2

#设置权重,偏置
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))

y=Weights*x_data+biases

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()
###激活
sess=tf.Session()
sess.run(init)

for step in range(300):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weights),sess.run(biases))