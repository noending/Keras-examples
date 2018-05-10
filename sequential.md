## Keras Sequential 模型方法

### 1.compile
---
```compile(self, optimizer, loss, metrics=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)```
用于配置训练模型。

#### 参数
* optimizer: 字符串（优化器名）或者优化器对象。详见 optimizers。
* loss: 字符串（目标函数名）或目标函数。详见 losses。 如果模型具有多个输出，则可以通过传递损失函数的字典或列表，在每个输出上使用不同的损失。模型将最小化的损失值将是所有单个损失的总和。
* metrics: 在训练和测试期间的模型评估标准。通常你会使用 metrics = ['accuracy']。 要为多输出模型的不同输出指定不同的评估标准，还可以传递一个字典，如 metrics = {'output_a'：'accuracy'}。
* sample_weight_mode: 如果你需要执行按时间步采样权重（2D权重），请将其设置为 temporal。 默认为 None，为采样权重（1D）。如果模型有多个输出，则可以通过传递 mode 的字典或列表，以在每个输出上使用不同的 sample_weight_mode。
* weighted_metrics: 在训练和测试期间，由 sample_weight 或 class_weight 评估和加权的度量标准列表。
---

#### 1.1 optimizer:
---
#### 1.1.1SGD
```keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)```
随机梯度下降优化器
包含扩展功能的支持： - 动量（momentum）优化, - 学习率衰减（每次参数更新后） - Nestrov动量(NAG)优化
#### 参数
* lr: float >= 0. 学习率
* momentum: float >= 0. 参数，用于加速SGD在相关方向上前进，并抑制震荡
* decay: float >= 0. 每次参数更新后学习率衰减值.
* nesterov: boolean. 是否使用Nesterov动量.

#### 1.1.2.RMSprop 

```keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)```
RMSProp优化器.
建议使用优化器的默认参数 （除了学习率lr，它可以被自由调节）
这个优化器通常是训练循环神经网络RNN的不错选择。
参数
* lr: float >= 0. 学习率.
* rho: float >= 0. RMSProp梯度平方的移动均值的衰减率.
* epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon().
* decay: float >= 0. 每次参数更新后学习率衰减值.

#### 1.1.3.Adagrad

keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
Adagrad优化器.
建议使用优化器的默认参数。
参数
* lr: float >= 0. 学习率.
* epsilon: float >= 0. 若为 None, 默认为 K.epsilon().
* decay: float >= 0. 每次参数更新后学习率衰减值.

#### 1.1.4.Adadelta

```keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)```
Adagrad优化器.
建议使用优化器的默认参数。
参数
* lr: float >= 0. 学习率，建议保留默认值.
* rho: float >= 0. Adadelta梯度平方移动均值的衰减率
* epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon().
* decay: float >= 0. 每次参数更新后学习率衰减值.

#### 1.1.5.Adam

```keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)```
Adam优化器.
默认参数遵循原论文中提供的值。
参数
* lr: float >= 0. 学习率.
* beta_1: float, 0 < beta < 1. 通常接近于 1.
* beta_2: float, 0 < beta < 1. 通常接近于 1.
* epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon().
* decay: float >= 0. 每次参数更新后学习率衰减值.
* amsgrad: boolean. 是否应用此算法的AMSGrad变种，来自论文"On the Convergence of Adam and Beyond".

---
#### 1.2 loss 损失函数（或称目标函数loss、优化评分函数optimizer）是编译模型时所需的两个参数之一 
``` eg: model.comple(loss='categorical_crossentropy', optimizer=‘Adam'…)```
#### 1.2.0. 常用损失函数 Losses
#### 1.2.1.mean_squared_error

```mean_squared_error(y_true, y_pred)```

#### 1.2.2.categorical_crossentropy

```categorical_crossentropy(y_true, y_pred)```

#### 1.2.3.binary_crossentropy

```binary_crossentropy(y_true, y_pred)```

注意: 当使用categorical_crossentropy损失时，你的目标值应该是分类格式 (即，如果你有10个类，每个样本的目标值应该是一个10维的向量，这个向量除了表示类别的那个索引为1，其他均为0)。 为了将 整数目标值 转换为 分类目标值，你可以使用Keras实用函数to_categorical：

from keras.utils.np_utils import to_categorical
categorical_labels = to_categorical(int_labels, num_classes=None)

---

### 2.fit

fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1,
callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

eg. model.fit(x_train,y_train,batch_size = 64, epochs = 10)

以固定数量的轮次（数据集上的迭代）训练模型。


#### 参数
* x: 训练数据的 Numpy 数组。 如果模型中的输入层被命名，你也可以传递一个字典，将输入层名称映射到 Numpy 数组。 如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，x 可以是 None（默认）。
* y: 目标（标签）数据的 Numpy 数组。 如果模型中的输出层被命名，你也可以传递一个字典，将输出层名称映射到 Numpy 数组。 如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，y 可以是 None（默认）。
* batch_size: 整数或 None。每次提度更新的样本数。如果未指定，默认为 32.
* epochs: 整数。训练模型迭代轮次。一个轮次是在整个 x 或 y 上的一轮迭代。请注意，与 initial_epoch 一起，epochs 被理解为 「最终轮次」。模型并不是训练了 epochs 轮，而是到第 epochs 轮停止训练。
* verbose: 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
* callbacks: 一系列的 keras.callbacks.Callback 实例。一系列可以在训练时使用的回调函数。详见 callbacks。
* validation_split: 在 0 和 1 之间浮动。用作验证集的训练数据的比例。模型将分出一部分不会被训练的验证数据，并将在每一轮结束时评估这些验证数据的误差和任何其他模型指标。验证数据是混洗之前 x 和y 数据的最后一部分样本中。
* validation_data: 元组 (x_val，y_val) 或元组 (x_val，y_val，val_sample_weights)，用来评估损失，以及在每轮结束时的任何模型度量指标。模型将不会在这个数据上进行训练。这个参数会覆盖 validation_split。
* shuffle: 布尔值（是否在每轮迭代之前混洗数据）或者 字符串 (batch)。batch 是处理 HDF5 数据限制的特殊选项，它对一个 batch 内部的数据进行混洗。当 steps_per_epoch 非 None 时，这个参数无效。
* class_weight: 可选的字典，用来映射类索引（整数）到权重（浮点）值，用于加权损失函数（仅在训练期间）。这可能有助于告诉模型 「更多关注」来自代表性不足的类的样本。
* sample_weight: 训练样本的可选 Numpy 权重数组，用于对损失函数进行加权（仅在训练期间）。您可以传递与输入样本长度相同的平坦（1D）Numpy 数组（权重和样本之间的 1：1 映射），或者在时序数据的情况下，可以传递尺寸为 (samples, sequence_length) 的 2D 数组，以对每个样本的每个时间步施加不同的权重。在这种情况下，你应该确保在 compile() 中指定 sample_weight_mode="temporal"。
* initial_epoch: 开始训练的轮次（有助于恢复之前的训练）。
* steps_per_epoch: 在声明一个轮次完成并开始下一个轮次之前的总步数（样品批次）。使用 TensorFlow 数据张量等输入张量进行训练时，默认值 None 等于数据集中样本的数量除以 batch 的大小，如果无法确定，则为 1。
* validation_steps: 只有在指定了 steps_per_epoch时才有用。停止前要验证的总步数（批次样本）。
返回
一个 History 对象。其 History.history 属性是连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录（如果适用）。
异常
* RuntimeError: 如果模型从未编译。
* ValueError: 在提供的输入数据与模型期望的不匹配的情况下。

---
### 3.evaluate

```evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)```
计算一些输入数据的误差，逐批次。
eg. model.evaluate(x_test,y_test)
#### 参数
* x: 输入数据，Numpy 数组或列表（如果模型有多输入）。 如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，x 可以是 None（默认）。
* y: 标签，Numpy 数组。 如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，y 可以是 None（默认）。
* batch_size: 整数。每次梯度更新的样本数。如果未指定，默认为 32。
* verbose: 日志显示模式，0 或 1。
* sample_weight: 样本权重，Numpy 数组。
* steps: 整数或 None。 声明评估结束之前的总步数（批次样本）。默认值 None。
返回
标量测试误差（如果模型没有评估指标）或标量列表（如果模型计算其他指标）。 属性 model.metrics_names 将提供标量输出的显示标签。
异常
* RuntimeError: 如果模型从未编译。

---
### 4.predict

```predict(self, x, batch_size=None, verbose=0, steps=None)```
为输入样本生成输出预测。
输入样本逐批处理。

#### 参数
* x: 输入数据，Numpy 数组。
* batch_size: 整数。如未指定，默认为 32。
* verbose: 日志显示模式，0 或 1。
* steps: 声明预测结束之前的总步数（批次样本）。默认值 None。
返回
预测的 Numpy 数组。

