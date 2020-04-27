'''
0.算法实现,以Python为例相关库函数
    - import numpy as np
        np.random.uniform(-1,1,(10,2)):生成10*2的数组,数组元素大小[-1,1)
        np.random.rand(10,2):生成10*2的数组,数组元素大小随机默认范围[0,1)
        np.sum(x,axis=1):axis=0,往里看0层,=1,往里看1层
        np.argmin(x):返回数组x中最小元素的下标
        np.mean():求均值
    - import random
        random.sample(list, 2):从列表中随机选择两个元素
        random.random():生成0和1之间的随机浮点数float
    - import matplotlib.pyplot as plt
        plt.figure():
        plt.show():
        plt.clf():清除所有轴,窗口打开,重复利用窗口的,已达到动画显示的效果

    - import networkx as nx
        G = nx.read_edgelist("twitter.txt",create_using=nx.Graph())  #读取文件，构建网络
        G.nodes()
        G.edges()
        G.neighbors('0') #注节点是"字符串"型

        networkx.classes.function.subgraph(G, nbunch)


    - import copy
        d=copy.deepcopy(alist)
'''

import networkx as nx
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from IMSC_functions import *

# 社团划分
# 输入:图G的文件名
# 输出:社区划分结果(划分结果用子图数组表示)


def community_detection(G):
    gs = label_propagation_community(G)  # 节点数组,返回图G的每个子图的节点集
    gs.sort(key=lambda x: len(x), reverse=True)  # 节点数组,按照子图节点的个数从大到小排序
    # 删除节点数少的子图的节点数组
    while len(gs[-1]) < 20:
        gs.pop()
    # 子图数组,子图按照节点数组来划分(诱导)产生子图
    Gs = []
    for i in gs:
        Gs.append(nx.classes.function.subgraph(G, i))
    return Gs


def DPSO(G, n, gmax, w, c1, c2, k):
    '''
    maxgen = 100  # 最大迭代次数 gmax
    w = 0.8  # 惯性权重 w
    c1 = 2.0  # 学习因子1 c1
    c2 = 2.0  # 学习因子二 c2
    K = 10  # 种子集大小 k
    pop = 100 # 粒子群大小 n
    '''




    # 开始时间
    start_time = time.perf_counter()

    # 图的节点集,按度从大到小排列
    all_nodes = sorted(list(G.nodes()),key=lambda x:len(list(G.neighbors(x))),reverse=True)


    # PSO算法的迭代过程

    # 种群初始化

    # 粒子位置初始化,每个粒子的位置向量数为k,共有n个粒子
    Position_s = population_initialization(G, all_nodes, n, k) # 粒子当前位置,集
    # 粒子速度初始化,全设置为0,每个粒子的速度向量数为k,共有n个粒子
    Velocity_s = [] # 粒子当前速度,集
    for _ in range(n):
        Velocity_s.append([])
    for i in range(n):
        for _ in range(k):
            Velocity_s[i].append(0)
    # 粒子历史最佳位置,集
    Pbest_s = Position_s
    Pbest_s_fitness = []  # 粒子历史最佳位置的适应度,集
    for i in Pbest_s:
        Pbest_s_fitness.append(EDV(G, i, k))

    # 全局最佳粒子的最佳位置
    Gbest = sorted(Pbest_s, key=lambda x: EDV(G, x, k), reverse=True)[0]
    Gbest_fitness = EDV(G, Gbest, k) # 全局最佳粒子的最佳位置的适应度


    # 代数
    g = 0
    while g < gmax:
        # 更新速度和位置
        for i in range(n):
            Position_s[i], Velocity_s[i] = update_Position(G, all_nodes, Pbest_s[i], Gbest, Velocity_s[i], Position_s[i], k, c1, c2, w)
        # 更新P_best
        for i in range(n):
            X_fitness = EDV(G,Position_s[i],k)
            if X_fitness > Pbest_s_fitness[i]:
                Pbest_s[i] = Position_s[i]
                Pbest_s_fitness[i] = X_fitness

        # 更新G_best
        Gbest_candidate = sorted(Pbest_s, key=lambda x: EDV(G, x, k), reverse=True)[0]
        Gbest_candidate_fitness = EDV(G, Gbest_candidate, k)
        Gbest_candidate = local_search(G, Gbest_candidate, Gbest_candidate_fitness, k)

        if Gbest_fitness < Gbest_candidate_fitness:
            Gbest = Gbest_candidate
            Gbest_fitness = Gbest_candidate_fitness
        g += 1
        # 结束时间
        end_time = time.perf_counter()

        runningtime = end_time - start_time

        C = []
        C.append(runningtime)
        print("gen:%d 时间:%d"%(g, round(np.mean(C), 1)))

    # 种子集
    return Gbest

def main():
    file_name = "hamster.txt"
    G = nx.read_edgelist(file_name, create_using=nx.Graph())  # 网络,读取文件构建网络
    GS = community_detection(G)

    n = 100 # 粒子群大小
    gmax = 100  # 最大迭代次数
    w = 0.8  # 惯性权重
    c1 = 2.0  # 学习因子1
    c2 = 2.0  # 学习因子2
    k = 10  # 种子集大小

    S = []  # 种子集
    print("网络:%s" % file_name)
    ii = 0
    for i in GS:
        ii += 1
        print("子网%d/%d:"%(ii,len(GS)))
        S.append(DPSO(i, n, gmax, w, c1, c2, k))
    S=sorted(S,key=lambda x: EDV(G, x, k), reverse=True)
    while len(S) > k:
        S.pop()
    diffusion_result_list_final = 0
    diffusion_result_list_final = IC_model(G, S)
    B = []
    B.append(diffusion_result_list_final)
    print("网络:%s 种子集大小:%d 影响力:%d"%(file_name ,k ,round(np.mean(B), 1)))

if __name__ == '__main__':
    main()
