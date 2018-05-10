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



