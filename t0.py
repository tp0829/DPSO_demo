# 笔记 2020年04月14日

import networkx as nx
import matplotlib.pyplot as plt

G_1 = nx.read_edgelist("twitter.txt",create_using=nx.Graph())  #读取文件，构建网络
G_2 = nx.read_edgelist("facebook.txt",create_using=nx.Graph())  #读取文件，构建网络
G_3 = nx.read_edgelist("hamster.txt",create_using=nx.Graph())  #读取文件，构建网络

len(G_1.nodes())
len(G_1.edges())

'''
1.数据集
    国外的数据集:
        小:twitter  节点:2939   边:15677
        大:facebook 节点:4039   边:88234
    国内的数据集:(暂无)
        小:
        大:

2.改进思路:
    - 数据量小的时候可以使用文中提出的DPSO算法
        - 通过"局部搜索"缩短了计算量
    - 如果数据量比较大,我们可否引入"社团划分"
        - 在进行DPSO之前,把我们数据集,通过社团划分,分成几个小的部分
        - 再分别对这几个小的部分进行DPSO操作.

3.实验设计方案:
    1) 参数的实验
        得出,合适的参数
    2) 引入社团划分有效性的实验
        引入社团划分的算法(DPSO-C)和没引入社团划分的算法(DPSO)
            分别在4个数据集(两个大的,两个小的)上进行实验
        分别得出,运行时间和影响力
    3) 对比分析
        分析运行时间和影响力大小
    4) 期望的结果是:
        (1) 影响力方面: DPSO-C 更好或和 DPSO 差不多
        (2) 运行时间方面:
            在小的数据集上,DPSO 时间更短
            在大的数据集上,DPSO-C 的优势将体现出来
        (3) 结论: 随着社交网络的越来越大,DPSO-C将有更好的前景
'''
