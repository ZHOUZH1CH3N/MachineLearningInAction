from numpy import *
import operator

def createDataSet () :
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k) :
    # 计算行数
    dataSetSize = dataSet.shape[0]
    # tile将原矩阵横向、纵向地复制，在这里纵向复制dataSetSize次
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat = diffMat**2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，计算出距离
    distances = sqDistances**0.5
    # 返回distances中元素从小到大排序后的索引值
    sortedDisIndicies = distances.argsort()
    # 定义一个记录类别次数的字典
    classCount = {}
    # 从0到k
    for i in range(k) :
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDisIndicies[i]]
        # dict.get(key,default=None)，字典的get()方法，返回指定键的值，如果值不存在字典中返回默认值
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # key=operator.itemgetter(1)根据字典的值进行排序
        # reverse降序排序字典
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别，即所要分类的类别
    return sortedClassCount[0][0]