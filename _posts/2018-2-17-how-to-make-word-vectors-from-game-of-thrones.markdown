---
layout: post
title: 词向量方法分析《权利的游戏》
date: 2018-02-17 20:12:33 +0800
categories: AI
tags: 深度学习 Siraj 
img: http://wangweiguang.xyz/images/game-of-thrones.jpg
---
* 
{:toc}
课程视频见：[https://www.bilibili.com/video/av19669011/](https://www.bilibili.com/video/av19669011/)

这周的主课讲了情感分析，这里讲解一些自然语言理解方面的程序例子。作为承载人类情感的载体，语言文字，常常作为情感分析的典型材料。所以情感分析和自然语言理解是密不可分的，他们都作为机器学习的一种应用领域，帮助人类深入思维的密境，揭示人类情感思维的奥秘。

只要输入足够的语料数据，并且运用一些自然语言处理的知识，就可以通过机器学习的方法解释文字中的种种关系和规律。这里，作为入门的例子，将用自然语言处理（NLP）的方式，用Python编程对著名外文的《冰与火之歌》进行简单的解析。

# 探索《权利的游戏》
## 依赖库

```python
# 导入python未来支持的语言特征，使python2环境下。可以使用python3的一些特性。
from __future__ import absolute_import, division, print_function

```

```python

#词汇的编码
import codecs
#快速搜索文件，找到匹配模式的所有路径名
import glob
#方便的输出详细的调试信息
import logging
#关于并发性的库，让程序更快
import multiprocessing
#处理操作系统相关东西，比如读文件
import os
#pretty print, 让输入更整洁美观
import pprint
#关于正则表达式
import re
```

```python
#自然语言处理库
import nltk
#转化词向量的库
import gensim.models.word2vec as w2v
#数据降维
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#可视化
import seaborn as sns
```

```python
#可以让 plot 出来的图片直接嵌入在notebook里面，而不用show
%pylab inline
```

```python
# 设置调试信息输出
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```

```python
# 下载NLTK里的分词器（仅需首次下载）
nltk.download("punkt")
nltk.download("stopwords")
```
## 准备语料库
### 导入

```python
# 导入书籍列表（五部）
book_filenames = sorted(glob.glob("data/*.txt"))
print("Found books:")
book_filenames
```

```
Found books:
Out[7]:
['data\\got1.txt',
 'data\\got2.txt',
 'data\\got3.txt',
 'data\\got4.txt',
 'data\\got5.txt']
```
### 合并

```python
# 合并所有文本为一个str
corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()
```

```
Reading 'data\got1.txt'...
Corpus is now 1770659 characters long

Reading 'data\got2.txt'...
Corpus is now 4071041 characters long

Reading 'data\got3.txt'...
Corpus is now 6391405 characters long

Reading 'data\got4.txt'...
Corpus is now 8107945 characters long

Reading 'data\got5.txt'...
Corpus is now 9719485 characters long
```

### 分句
```python
# 将语料分成一个个单句
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)
```

### 分词&清洗

```python
#在把句子进行分词
#简单清洗语料，去除a-z，A-Z之外的字符
#list of words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words
```

```python
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))
```

效果展示：

```python
print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))
```

```
Heraldic crest by Virginia Norey.
['Heraldic', 'crest', 'by', 'Virginia', 'Norey']
```

```python
token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))
```

```
The book corpus contains 1,818,103 tokens
```

## 训练Word2Vec模型
### 初始化参数
```python
#ONCE we have vectors
#step 3 - build model
#3 main tasks that vectors help with
#DISTANCE, SIMILARITY, RANKING

# Dimensionality of the resulting word vectors.
#more dimensions, more computationally expensive to train
#but also more accurate
#more dimensions = more generalized
num_features = 300
# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1
```

```python
thrones2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
```

### 建立词汇表
```python
thrones2vec.build_vocab(sentences)
```

```python
print("Word2Vec vocabulary length:", len(thrones2vec.wv.vocab))
```

```
Word2Vec vocabulary length: 17277
```

### 开始训练
```python
thrones2vec.train(sentences,total_examples=thrones2vec.corpus_count,epochs=5)
```

### 保存模型
训练完将模型保存下来，不然关了在开又得再训练4个小时,,,
```python
if not os.path.exists("trained"):
    os.makedirs("trained")

thrones2vec.save(os.path.join("trained1", "thrones2vec.w2v"))
```


## 模型探索
### 载入模型
```python
thrones2vec = w2v.Word2Vec.load(os.path.join("trained", "thrones2vec.w2v"))
```

### 模型可视化
Siraj Raval有一个专门的课程讲数据分析和可视化的，面对大型的多维的数据如何可视化到一张图上，这里先作为预习，后天再详细学习一下。

这里是把《权利的游戏》上面这17277个词汇进行了一种相关度的分析，这些词有人物，有物品，也有地方等。接下来就把全部词汇可视化到一个二维平面上（需要降维），每个词作为图上一个点，点与点之间的距离的远近表示这两个关系的大小。

```python
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
```

```python
all_word_vectors_matrix = thrones2vec.wv.vectors
```

```python
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
```

```python
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[thrones2vec.wv.vocab[word].index])
            for word in thrones2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)
```

```python
points.head(10)
```
输出前十个点（词和坐标）
```
	word	x	y
0	fawn	-4.470860	-0.406855
1	raining	2.432409	-1.825349
2	writings	-3.212095	1.967637
3	Ysilla	1.436866	-2.421560
4	Rory	-1.090941	-2.569549
5	hordes	-2.204853	2.614524
6	mustachio	-1.086925	-3.887781
7	Greyjoy	1.585396	3.667034
8	yellow	-0.813293	-5.425221
9	four	1.871287	2.557694

```

```python

sns.set_context("poster")
```

```python
points.plot.scatter("x", "y", s=10, figsize=(20, 12))
```
输出整幅图：
（很壮观呀，有一点像狼头）
![image](http://wangweiguang.xyz/images/plot.jpg)

```python
def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
```

放大观察图的指定区域：
```python
plot_region(x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))
```
与御林铁卫有关的角色最终走到了一起。
![image](http://wangweiguang.xyz/images/plot1.jpg)

```python
plot_region(x_bounds=(0, 1), y_bounds=(4, 4.5))
```
食物有关的词也很好的组合在一起。
![image](http://wangweiguang.xyz/images/plot2.jpg)

## 小说解析
小说我没看过，不过电视剧大概看完了，下面简单分析一下模型得出来的结果。

与“Stark”最相关的点：
```python
thrones2vec.wv.most_similar("Stark")
```

```
[(u'Eddard', 0.742438018321991),
 (u'Winterfell', 0.64848792552948),
 (u'Brandon', 0.6438549757003784),
 (u'Lyanna', 0.6438394784927368),
 (u'Robb', 0.6242259740829468),
 (u'executed', 0.6220564842224121),
 (u'Arryn', 0.6189971566200256),
 (u'Benjen', 0.6188897490501404),
 (u'direwolf', 0.614366352558136),
 (u'beheaded', 0.6046538352966309)]
```
1. Eddard：与“史塔克”最相关的人无疑还是第一任城主奈德·斯塔克了，他为了国家和史塔克家族的名誉尽心尽力，乃至付出了生命，最终的却是在人民的唾骂声中人头落地。
2. Winterfell：临冬城，史塔克家族的老家。
3. Brandon：布兰·史塔克，临冬城公爵艾德·史塔克和凯特琳夫人的第三子，艾莉亚之弟。
4. Lyanna：，，，这是谁呀，似乎没印象，，
5. Robb：艾德·史塔克的长子，第二任城主好像是，记得史塔克家族似乎是覆灭在他手上了，，，
6. executed：这是执行死刑的意思吧，对与史塔克家族杯具的命运也是不幸的说对了。


与“Aerys”（疯王）最相关的点：
```python
thrones2vec.wv.most_similar("Aerys")
```

```
[(u'Jaehaerys', 0.7991689443588257),
 (u'Daeron', 0.7808291912078857),
 (u'II', 0.7649893164634705),
 (u'reign', 0.7466063499450684),
 (u'Mad', 0.7380156517028809),
 (u'Beggar', 0.7334001660346985),
 (u'Rhaegar', 0.7308052182197571),
 (u'Unworthy', 0.7120681405067444),
 (u'Cruel', 0.7089171409606934),
 (u'Dome', 0.7070454359054565)]
```

“疯王”我感觉在剧中一直是谜一样的人物，与他相关的点是：
1. Jaehaerys：查了一下，一个神秘的人物，全剧都没有直接介绍过，据说是雪诺的真名？和疯王有什么关系？
2. ​

与“direwolf”最相关的点：
```python
thrones2vec.wv.most_similar("direwolf")
```
```
[(u'Rickon', 0.6617892980575562),
 (u'SHAGGYDOG', 0.643834114074707),
 (u'wolf', 0.6403605341911316),
 (u'GHOST', 0.6385751962661743),
 (u'pup', 0.6156360507011414),
 (u'Robb', 0.6147520542144775),
 (u'Stark', 0.614366352558136),
 (u'crannogman', 0.6082616448402405),
 (u'wight', 0.606614351272583),
 (u'RICKON', 0.6039268970489502)]
```
1. Rickon：史塔克夫人的儿子，印象中和他的冰原狼“毛毛狗”一直是形影不离的。
2. SHAGGYDOG：“毛毛狗”

```python
def nearest_similarity_cosmul(start1, end1, end2):
    similarities = thrones2vec.wv.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2
```
最后利用词汇间的相关性，造一些有趣的句子。
```python
nearest_similarity_cosmul("Stark", "Winterfell", "Riverrun")
nearest_similarity_cosmul("Jaime", "sword", "wine")
nearest_similarity_cosmul("Arya", "Nymeria", "dragons")
```
```
Stark is related to Winterfell, as Tully is related to Riverrun
Jaime is related to sword, as Tyrion is related to wine
Arya is related to Nymeria, as Dany is related to dragons
u'Dany
```

## 总结
这次的课程还是相当有意思的，想对于上面《权利的游戏》机器学习模型，除了再说一遍简单的相关性分析之外，还有好多其他的点子可以玩。这个程序完成折腾下来费的功夫也是不小了，出了有好多函数的用法已经被弃用之外，“gensim”库还是没有完全安好，提示的警告是“C extension not loaded, training will be slow.”，不过好歹运行了4、5个小时训练出来了，以后有机会在弄吧。自然语言理解这部分还是不过瘾，接下来我得找一篇中文的小说分析一下。