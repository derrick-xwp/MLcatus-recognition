#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:49:56 2020

@author: xingwenpeng
"""


import pandas as pd
import os,shutil,math,scipy,cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rn


from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,roc_curve,auc

from PIL import Image
from PIL import Image as pil_image
from PIL import ImageDraw

from time import time
from glob import glob
from tqdm import tqdm
from skimage.io import imread
from IPython.display import SVG

from scipy import misc,ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread


from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras import layers
from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler

def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    
augs_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2, #浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
    horizontal_flip=True, #布尔值，进行随机水平翻转
    vertical_flip=True)  #布尔值，进行随机竖直翻转
#生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的epoch次数为止。
#keras.preprocessing.image模块中的图片生成器，同时也可以在batch中对数据进行增强，扩充数据集大小，增强模型的泛化能力。
#比如进行旋转，变形，归一化等等。
train_gen = augs_gen.flow_from_directory(
    '/Users/xingwenpeng/Desktop/cactus-aerial-photos/training_set/training_set',
    target_size = (32,32),
    batch_size=32,
    shuffle=True,
    class_mode = 'binary'
)

test_gen = augs_gen.flow_from_directory(
    '/Users/xingwenpeng/Desktop/cactus-aerial-photos/validation_set/validation_set/',
    target_size=(32,32),
    batch_size=32,
    shuffle=False,
    class_mode = 'binary'
)  
model = Sequential()
model.add(Conv2D(6, kernel_size=(5,5),activation='relu',input_shape=(32,32,3)))
model.add(BatchNormalization())
#上层网络需要不停调整来适应输入数据分布的变化，导致网络学习速度的降低
#网络的训练过程容易陷入梯度饱和区，减缓网络收敛速度
#有两种解决思路。第一种就是更为非饱和性激活函数，例如线性整流函数ReLU可以在一定程度上解决训练进入梯度饱和区的问题。
#另一种思路是，我们可以让激活函数的输入分布保持在一个稳定状态来尽可能避免它们陷入梯度饱和区，这也就是Normalization的思路。
model.add(MaxPooling2D(pool_size=2,strides=2))
#pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
#strides：整数或长为2的整数tuple，或者None，步长值。
model.add(Conv2D(16,kernel_size=5,strides=1,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2,strides=2))
#。。。。。。。
'''
model.add(Conv2D(32,kernel_size=5,strides=1,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2,strides=2))
'''
#。。。。。。。

model.add(Flatten())
#Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
model.add(Dense(120,activation='relu'))

model.add(Dense(84,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.5))
#正则化减少过拟合。
#在训练过程中随机将该层的一些输出特征舍弃（设置为0）。
# dropout 比率（dropout rate）是被设为 0 的特征所占的比例，通常在 0.2~0.5范围内。测试时没有单元被舍弃，而该层的输出值需要按 dropout 比率缩小。
'''
防止神经网络过拟合的常用方法

    获取更多的训练数据
    减小网络容量
    添加权重正则化
    添加 dropout

'''

model.add(Dense(1,activation='sigmoid'))
#全连接层
#units：大于0的整数，代表该层的输出维度。
#activation：激活函数，为预定义的激活函数名（参考激活函数）、如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
model.summary()


SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#使用回调函数来查看训练模型的内在状态和统计。在每个训练期之后保存模型。
best_model_weights = './base.model'
checkpoint = ModelCheckpoint(
    best_model_weights,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)
'''
1）loss一直下降，收敛，而val_loss却上升，不收敛，说明过拟合了。loss下降、收敛，说明模型在训练集上，表现良好，但是却在验证集、测试集上没有良好的表现，这就是典型过拟合现象。

2）loss、val_loss一同上升、收敛：模型表现良好。
参数

    filepath: 字符串，保存模型的路径。
    monitor: 被监测的数据。
    verbose: 详细信息模式，0 或者 1 。
    save_best_only: 如果 save_best_only=True， 被监测数据的最佳模型就不会被覆盖。
    mode: {auto, min, max} 的其中之一。 如果 save_best_only=True，那么是否覆盖保存文件的决定就取决于被监测数据的最大或者最小值。 对于 val_acc，模式就会是 max，而对于 val_loss，模式就需要是 min，等等。 在 auto 模式中，方向会自动从被监测的数据的名字中判断出来。
    save_weights_only: 如果 True，那么只有模型的权重会被保存 (model.save_weights(filepath))， 否则的话，整个模型会被保存 (model.save(filepath))。
    period: 每个检查点之间的间隔（训练轮数）。
'''
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode='auto'
)
'''
当被监测的数量不再提升，则停止训练。
    monitor: 被监测的数据。
    min_delta: 在被监测的数据中被认为是提升的最小变化， 例如，小于 min_delta 的绝对变化会被认为没有提升。
    patience: 没有进步的训练轮数，在这之后训练就会被停止。
    verbose: 详细信息模式。
    mode: {auto, min, max} 其中之一。 在 min 模式中， 当被监测的数据停止下降，训练就会停止；在 max 模式中，当被监测的数据停止上升，训练就会停止；在 auto 模式中，方向会自动从被监测的数据的名字中判断出来。
    baseline: 要监控的数量的基准值。 如果模型没有显示基准的改善，训练将停止。
    restore_best_weights: 是否从具有监测数量的最佳值的时期恢复模型权重。 如果为 False，则使用在训练的最后一步获得的模型权重。

'''
tensorboard = TensorBoard(
    log_dir = './logs',
    histogram_freq=0,
    batch_size=16,
    write_graph=True,
    write_grads=True,
    write_images=False,
)
'''
这个回调函数为 Tensorboard 编写一个日志， 这样你可以可视化测试和训练的标准评估的动态图像， 也可以可视化模型中不同层的激活值直方图。
'''
csvlogger = CSVLogger(
    filename= "training_csv.log",
    separator = ",",
    append = False
)
'''
把训练轮结果数据流到 csv 文件的回调函数。

    filename: csv 文件的文件名，例如 'run/log.csv'。
    separator: 用来隔离 csv 文件中元素的字符串。
    append: True：如果文件存在则增加（可以被用于继续训练）。False：覆盖存在的文件。

'''
#lrsched = LearningRateScheduler(step_decay,verbose=1)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1, 
    mode='auto',
    cooldown=1 
)
'''
    monitor: 被监测的数据。
    factor: 学习速率被降低的因数。新的学习速率 = 学习速率 * 因数
    patience: 没有进步的训练轮数，在这之后训练速率会被降低。
    verbose: 整数。0：安静，1：更新信息。
    mode: {auto, min, max} 其中之一。如果是 min 模式，学习速率会被降低如果被监测的数据已经停止下降； 在 max 模式，学习塑料会被降低如果被监测的数据已经停止上升； 在 auto 模式，方向会被从被监测的数据中自动推断出来。
    min_delta: 对于测量新的最优化的阀值，只关注巨大的改变。
    cooldown: 在学习速率被降低之后，重新恢复正常操作之前等待的训练轮数量。
    min_lr: 学习速率的下边界。

'''
callbacks = [checkpoint,tensorboard,csvlogger,reduce]

opt = SGD(lr=2e-4,momentum=0.99)
'''
随机梯度下降法，支持动量参数，支持学习衰减率，支持Nesterov动量
参数

    lr：大或等于0的浮点数，学习率

    momentum：大或等于0的浮点数，动量参数

    decay：大或等于0的浮点数，每次更新后的学习率衰减值

    nesterov：布尔值，确定是否使用Nesterov动量
'''

opt1 = Adam(lr=1e-2)

'''
    lr：大或等于0的浮点数，学习率

    beta_1/beta_2：浮点数， 0<beta<1，通常很接近1

    epsilon：大或等于0的小浮点数，防止除0错误

'''

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
'''
optimizer：优化器，如Adam

loss：计算损失，这里用的是交叉熵损失
　　目标函数，或称损失函数，是网络中的性能函数，也是编译一个模型必须的两个参数之一
   即对数损失函数，log loss，与sigmoid相对应的损失函数。
metrics: 列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=[‘accuracy’]。如果要在多输出模型中为不同的输出指定不同的指标，可向该参数传递一个字典，例如metrics={‘output_a’: ‘accuracy’}

loss下降，val_loss下降：训练网络正常，最好情况。

loss下降，val_loss稳定：网络过拟合化，可以使用正则化和Max pooling。

loss稳定，val_loss下降：数据集有严重问题，建议重新选择。

loss稳定，val_loss稳定：学习过程遇到瓶颈，需要减小学习率或批量数目，可以减少学习率。

loss上升，val_loss上升：网络结构设计问题，训练超参数设置不当，数据集经过清洗等问题，最差情况。

'''
    
history = model.fit_generator(
    train_gen, 
    validation_data = test_gen,
    validation_steps = 100,
    steps_per_epoch  = 100, 
    epochs = 100,
    verbose = 1,
    callbacks=callbacks
)
show_final_history(history)

fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax[1].set_title('acc')
ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
ax[0].legend()
ax[1].legend()




model.load_weights(best_model_weights)
model_eval = model.evaluate_generator(test_gen,steps=100)
print("Model Test Loss:",model_eval[0])
print("Model Test Accuracy:",model_eval[1])

model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
    
model.save("model.h5")
print("Weights Saved")

'''
When the features of the input data are significantly different, 
the effect of tanh is very good, 
and the feature effect will continue to expand and display in the process of circulation. 
When the difference of features is not obvious, 
the effect of sigmoid is better.
 At the same time, when sigmoid and tanh are used as activation functions, 
 the input needs to be normalized, otherwise, 
 all the activated values will enter the flat area,
 and the output of the hidden layer will all converge, 
 losing the original feature expression. Relu is much better, 
 and sometimes you don't need input normalization to avoid this. 
 Therefore, most convolutional neural networks use relu as activation function.
'''



'''
1. Dataset expansion
It can be considered to increase the capacity of data sets. Sometimes it is simple to increase the capacity, and the accuracy is improved significantly

2. Increase the difference and randomness of data set
When making data sets, you can consider increasing the difference of data

3.tensor transform
Using transform module to process data in pytorch

4. Batch size
Adjust the size of batch_, you can adjust 16, 32, 64... In this way, you can go up one by one to find the most suitable one, of course, you don't need to be a multiple of 2

5.shuffle=True
Random reading data, generally used in network training, is amazing for small data sets

6.learning rate
Can use dynamic learning rate
You can also start with a large learning rate, and then slowly reduce it. For example, at the beginning, it is 0.1, and then it is 0.05. Each time, half points are given to find the most appropriate learning rate

7.weight_decay
Weight attenuation is also called L2 regularization. You can do it yourself. Hahahaha

8.n_epochs
When the accuracy no longer increases, the number of learning can be increased to make the network fully learn. Of course, over fitting should be prevented,

9. Dropout() parameter and location
Dropout() parameters can be adjusted from 0.1 to 0.05. Each network has different parameters
The location also has a great influence, not necessarily after the full connection layer

10. Parameter initialization
Default parameter initialization in Python
If not, consider self initialization to help improve accuracy

11. Network structure (number of layers, size of each layer, function)
Consider increasing levels, or resizing the output to improve accuracy



'''
