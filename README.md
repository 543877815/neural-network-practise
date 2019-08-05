# 深度学习之神经网络（CNN RNN GAN）算法原理+实战练习

并没有全部搞懂，写给自己以后看

## 虚拟环境的配置

见[我的博客](https://543877815.github.io/2019/07/06/%E4%BB%8Epython%E5%BC%80%E5%A7%8B%E7%9A%84%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE%E6%97%A5%E5%BF%97/)

## 1. 分类任务

### 数据

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

数据说明详情见[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

数据类型例子见[data introduction.ipynb](./data introduction.ipynb)

### 实现

#### 逻辑斯蒂二分类

对类别0和1进行分类，实现见[neuron.ipynb](./neuron.ipynb)

100000次训练

数据未归一化测试集准确率51%

数据归一化后测试集准确率82%

#### 单层神经网络

逻辑斯蒂，softmax多分类

实现见[neuron-multi_output.ipynb](./neuron-multi_output.ipynb)

100000次训练，测试集准确率31.6%

#### 多层神经网络

实现见[neural-network.ipynb](./neural-network.ipynb)

10000次训练，测试集准确率46.4%

#### 卷积神经网络

实现见[convnet.ipynb](./convnet.ipynb)

10000次训练，测试集准确率66.3%

#### vgg-net

实现见[vgg-net.ipynb](./vgg-net.ipynb)

10000次训练，测试集准确率73.2%

#### res-net

实现见[res-net.ipynb](./res-net.ipynb)

10000次训练，测试集准确率76.3%

#### inception-net

实现见[inception-net.ipynb](./inception-net.ipynb)

10000次训练，测试集准确率73.5%

#### mobile-net

实现见[mobile-net.ipynb](./mobile-net.ipynb)

10000次训练，测试机准确率64.9%

### 可视化学习与迁移学习

#### tensorboard

实现见[vgg-tensorboard.ipynb](./vgg-tensorboard.ipynb)

```bash
tensorboard --logdir=train:'[train-path]',test:'[test-path]'
```

#### fine-tune

实现见[vgg-tensorboard-fine-tune.ipynb](./vgg-tensorboard-fine-tune.ipynb)

1. save models (third party/myself)
2. restore models checkpoints (断点恢复)
3. keep some layers fixed.

### 调参

#### activation-initializer-optimizer

实现见[vgg-tensorboard-activation-initializer-optimizer.ipynb](./vgg-tensorboard-activation-initializer-optimizer.ipynb)

修改的参数列表

- activation: relu, sigmoid, tanh

- weight initializer: he, xavier, normal, truncated_normal

- optimizer: Adam, Momentum, Gradient Descent.

一些效果：  

- flatten = convnet(x_image, tf.nn.relu) # train 10k: *73.9%*

- flatten = convnet(x_image, tf.nn.relu, None)  # train *74.8%* 100k train

- flatten = convnet(x_image, tf.nn.relu, tf.truncated_normal_initializer(stddev=0.02))  # *72.9%* 100k train

- flatten = convnet(x_image, tf.nn.relu, tf.keras.initializers.he_normal)  # *73.1%* 100k train

- flatten = convnet(x_image, tf.nn.sigmoid) # train 10k: *69.05%*

- train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss) # *44.20%* train 100k

- train_op = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9).minimize(loss) *71.8%* train 100k

#### data augmentation

实现见[vgg-tensorboard-data_aug.ipynb](./vgg-tensorboard-data_aug.ipynb)

- data_aug_1 = tf.image.random_flip_left_right(x_single_image)
- data_aug_2 = tf.image.random_brightness(data_aug_1, max_delta=65)
- data_aug_3 = tf.image.random_contrast(data_aug_2, lower=0.2, upper=1.8)

train10000次，测试集准确率68.25%

train100000次，测试集准确率78.05%

#### deeper layers

实现见[vgg-tensorboard-data_aug_deeper-bn.ipynb](./vgg-tensorboard-data_aug_deeper-bn.ipynb)

train 100000次，测试集准确率73.55%

#### batch normalization

实现见[vgg-tensorboard-data_aug_deeper-bn.ipynb](./vgg-tensorboard-data_aug_deeper-bn.ipynb)

train 100000次，测试机准确率82.450%

## 2. 图像风格转换

### 数据

预训练模型VGG16，[github](https://github.com/machrisaa/tensorflow-vgg)，download from [here](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM)

数据格式见[vgg16_content.ipynb](./vgg16_content.ipynb)

### 实现

实现见[style_transfer.ipynb](./style_transfer.ipynb)

原始图片见[run_style_transfer](/run_style_transfer)

效果见[run_style_transfer](/run_style_transfer)（没有调参所以结果一般）

## 3.文本分类

### 数据

链接：https://pan.baidu.com/s/1GdRF6th_N2L7TVaC8YYTSg 提取码：uraj 
