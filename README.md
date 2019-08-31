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

### 预处理

[pre-processing.ipynb](./pre-processing.ipynb)

### text-rnn

实现见[text-rnn.ipynb](/text-rnn.ipynb)

架构：

```python
# 构建计算图——LSTM模型
#      embedding
#      LSTM
#      fc
#      train_op
# 训练流程代码
# 数据集封装
#      api: next_batch(batch_size)
# 词表封装
#      api: sentence2id(text_sentence): 句子转换id
# 类别封装
#      api: category2id(text_category)
```

训练10000次准确率：

Train: 99.7%

Valid: 92.7%

Test:  93.2%

### LSTM网络结构

实现见[text-rnn-pure-lstm.ipynb](./text-rnn-pure-lstm.ipynb)

keycode：

```python
with tf.variable_scope('lstm_nn', initializer = lstm_init):
    """
    cells = []
    for i in range(hps.num_lstm_layers):
        cell = tf.contrib.rnn.BasicLSTMCell(
            hps.num_lstm_nodes[i],
            state_is_tuple = True)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell,
            output_keep_prob = keep_prob)
        cells.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    
    initial_state = cell.zero_state(batch_size, tf.float32)
    # rnn_outputs: [batch_size, num_timesteps, lstm_outputs[-1]]
    rnn_outputs, _ = tf.nn.dynamic_rnn(cell, embed_inputs, initial_state = initial_state)
    last = rnn_outputs[:, -1, :]
    """
    # 输入门
    with tf.variable_scope('inputs'):
        ix, ih, ib = _generate_params_for_lstm_cell(
            x_size = [hps.num_embedding_size, hps.num_lstm_nodes[0]],
            h_size = [hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
            bias_size = [1, hps.num_lstm_nodes[0]]
        )
        
    # 输出门
    with tf.variable_scope('outputs'):
        ox, oh, ob = _generate_params_for_lstm_cell(
            x_size = [hps.num_embedding_size, hps.num_lstm_nodes[0]],
            h_size = [hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
            bias_size = [1, hps.num_lstm_nodes[0]]
        )
    
    # 遗忘门
    with tf.variable_scope('forget'):
        fx, fh, fb = _generate_params_for_lstm_cell(
            x_size = [hps.num_embedding_size, hps.num_lstm_nodes[0]],
            h_size = [hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
            bias_size = [1, hps.num_lstm_nodes[0]]
        )
    
    # 中间层
    with tf.variable_scope('memory'):
        cx, ch, cb = _generate_params_for_lstm_cell(
            x_size = [hps.num_embedding_size, hps.num_lstm_nodes[0]],
            h_size = [hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
            bias_size = [1, hps.num_lstm_nodes[0]]
        )
    state = tf.Variable(
        tf.zeros([batch_size, hps.num_lstm_nodes[0]]),
        trainable = False
    )
    h = tf.Variable(
        tf.zeros([batch_size, hps.num_lstm_nodes[0]]),
        trainable = False
    )
    for i in range(num_timesteps):
        # [batch_size, 1, embed_size]
        embed_input = embed_inputs[:, i, :]
        embed_input = tf.reshape(embed_input, [batch_size, hps.num_lstm_nodes[0]])
        forget_gate = tf.sigmoid(tf.matmul(embed_input, fx) + tf.matmul(h, fh) + fb)
        input_gate = tf.sigmoid(tf.matmul(embed_input, ix) + tf.matmul(h, ih) + ib)
        output_gate = tf.sigmoid(tf.matmul(embed_input, ox) + tf.matmul(h, oh) + ob)
        mid_state = tf.tanh(tf.matmul(embed_input, cx) + tf.matmul(h, ch) + cb)
        state = mid_state * input_gate + state * forget_gate
        h = output_gate * tf.tanh(state)
    last = h
```
### text-cnn

实现见[text-cnn.ipynb](./text-cnn.ipynb)

训练10000次准确率：

Train: 100%

Valid: 95.7%

Test:  95.2%

## 4.图像生成文本

### 数据集

[flickr30k_images](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html)

### 模型

[inception v3(http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz)](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz)

### 词表生成

见[generate_vocab.ipynb](./generate_vocab.ipynb)

### 训练

见[image_caption_train.ipynb](./image_caption_train.ipynb)

## 5.对抗生成网络

### 数据集

tensorflow 自带MNIST手写数字数据集

### 模型

[dc_gan.ipynb](./dc_gan.ipynb)

### 结果

训练10000次见[local_run](./local_run)

