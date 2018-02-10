录播视频中小作业：

1.BPTT(循环神经网络)

请看BPTT.py

BPTT的要点：

①reversed

②dhnext = dhraw

2.RNN中num_steps和state_size对学习效果的影响(RNN的Tensorflow实现)

视频中有3个文件，请查看raw_rnn_in_tf.ipynb、simple_rnn_in_tf.ipynb、dynamic_rnn_in_tf.ipynb
分别讲解了RNN的原始实现、Tensorflow简单封装实现和动态实现

关于num_steps和state_size对学习效果的影响：

实验结果：
id   num_steps    state_size    learns
1       10              16           2
2       5               16           1-2
3       3               16           不理想
4       5               4            1

实验分析：
num_steps代表着信息“流动”的距离，如果太短，网络学不到不同时间点上数据的关联关系，如实验3
state_size代表着网络对信息的承载能力，如果太小，则无法表达信息之间的关系，如实验5


HomeWork_10作业

1.对于Embedding的理解：

Why：

人类把自然语言交给计算机处理，首先要解决自然语言的数字化工作，Embedding提供了一种方式。

What:

Embedding可以看做是数学上的一个空间映射(mapping):map(lambda y:f(x)),该映射具有injective(单射)和structure-preserving(结构保持)两种特性。
对于WordEmbedding就是寻找一个mapping，将Word映射到空间上来表达。

How:

①One-hot Representation:稀疏的表达方式，将Word映射到坐标轴上，存储简洁，但是表达的信息量相对较少；而且由于过于稀疏容易产生维度爆炸；其次是无法表达Word之间的关系

②Distributed Representation(由Hinton在1986年提出)：稠密的表达方式，给one-hot降维，是低维实数向量，将Word映射到多维空间，不仅仅是坐标轴上。
这种表达方式相对信息量更丰富，不会产生维度爆炸，最重要的是可以通过欧氏距离或者cos夹角来体现Word之间的关系。其中经典的算法是word2vec

Extension:image embedding, video embedding


2.实验代码请看word2vec_basic_update.py(由word2vec_basic.py改编而来)
原理分析：
语义本质：它可以描述两个概念之间的语义关系，而这种语义关系完全是通过文档样本的学习来实现的，它不要求有任何对现实世界的语义建模输入（例如何为国家、国力、接壤等）。
一方面在现阶段进行常识建模的计算量非常大以至于不切合实际，另外也说明足够量的样本已经可以暴露出蕴含在其中的深层次语义概念。


实验分析：
词汇的语义相似度，由其对应向量的余弦相似度表示。因此在目标空间中，相似的词汇其向量将聚集为一处，如：百十三五八、东西南北等；
因为维度较高，所以向量对空间的填充密集度很小，模型的敏感度较高。

3.RNN训练

RNN的理解：

用一个权重矩阵W将不同时间点的深度神经网络(权重相同)“串”起来，其中隐层激活之前的logits由2部分相加而来：本层输入X*权重U + 上一时间点隐层激活后的输出*时间维度权重W
这个W在时间维度上是各个深度神经网络共享的。相应的，BP的时候，本层hidden-state的delta也由2部分组成：(上一层delta*V + 上一时间点delta*W)*本节点的激活函数导数 

RNN训练心得：

刚开始可以把learning_rate调成0.003，loss快速下降后，如果发生跳跃，则调为0.001，如果后面一直跳很难收敛，则调为0.0007，0.0003,0.0001

输出结果分析：

因为喂给网络的数据x是每个字，而label是向后错位一个字，x和错一位的label是一一对应的，比如x如果是“老夫聊发少年”，则label就是“夫聊发少年狂”
所以训练出来的网络能够记住每个字错一位后是哪个字


而sample.py中以title最后一个字为引子，连续取64个最可能下一个字，组成一首诗，由于语料库少，最终结果只有一定的语义信息，不能算作好的诗作。


遗留问题：

1.19220是怎么算出来的？我算的是1903073*30/128/32=13938 错在哪里了

2.sample.py中循环遍历title的作用是什么？
