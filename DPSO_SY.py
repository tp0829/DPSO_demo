# 离散粒子群优化算法,DPSO
# 学习笔记
'''
3.2 Proposed algorithm//算法
    - Algorithm 1 函数
        function Max(Gbest , Gbest)
            //选择LIE值较大的粒子
        Degree_Based_Initialization()
            //初始化函数
        Local_Search()
            //局部搜索
    - Algorithm 1 步骤
        输入:
            1.G //图G=(N,E,W)
            2.n //粒子群的大小
            3.gmax  //最大迭代次数
            4.w //惯性权重
            5.c1,c2 //学习因子
            6.k //种子集的大小
        第一步:
            初始化
                position vector:
                    X ← Degree_Based_Initialization(G,n,k);
                Pbest vector:
                    Pbest ← Degree_Based_Initialization(G,n,k);
                velocity vector:
                    V ← 0;
        第二步:
            选择全局最佳位置向量 Gbest
                According to the fitness value
        第三步:
            开始循环,设置g=0;
                Step3.1 Update V;
                Step3.2 Update X;
                Step3.3 Update the Pbest
                    and select out the best particle Gbest∗ in the current iteration;
                Step3.4 Employ the local search operation on Gbest∗: Gbest′ ← Local_Search(Gbest∗); Step3.5 Update the Gbest: Gbest ← Max(Gbest′, Gbest)
        第四步:
            for(g=0;g<gmax;g++)
        输出:
            best position Gbest as the seed set S.

3.0 算法实现,以Python为例相关库函数
    - import networkx as nx
        G = nx.read_edgelist("twitter.txt",create_using=nx.Graph())  #读取文件，构建网络
        G.nodes()
        G.edges()
        G.neighbors('0') #注节点是"字符串"型

    - import numpy as np
        np.random.uniform(-1,1,(10,2)):生成10*2的数组,数组元素大小[-1,1)
        np.random.rand(10,2):生成10*2的数组,数组元素大小随机默认范围[0,1)
        np.sum(x,axis=1):axis=0,往里看0层,=1,往里看1层
        np.argmin(x):返回数组x中最小元素的下标
        np.mean():求均值
    - import matplotlib.pyplot as plt
        plt.figure():
        plt.show():
        plt.clf():清除所有轴,窗口打开,重复利用窗口的,已达到动画显示的效果
    - import random
        random.sample(list, 2):从列表中随机选择两个元素
        random.random():生成0和1之间的随机浮点数float


'''


#!/usr/bin/env python
# -*- coding:utf-8 -*-
import networkx as nx
import time
import numpy as np
import random
import matplotlib.pyplot as plt

def IC_model(g,S,mc=10000,p=0.01):
    spread = []
    for _ in range(mc): # i
        # 模拟传播过程
        new_active, Au = S[:], S[:]
        while new_active:
            new_ones = []
            for node in new_active:
                nbs = list(set(g.neighbors(node)) - set(Au))
                for nb in nbs:
                    if random.random() <= p:
                        new_ones.append(nb)
            new_active = list(set(new_ones))
            Au += new_active
        spread.append(len(Au))
    return np.mean(spread)


## X 中保存 由种子集定义的 粒子位置
def SHD(G,all_nodes,K):
    X = [ ]

    candidate = sorted(all_nodes,key=G.nodes,reverse=True)
    X = candidate[:K]
    for ele_1 in range(K):
        if random.random() > 0.5:
            for ele_2 in random.sample(set(all_nodes)-set(X),1):
                X[ele_1] = ele_2

    return X

## pop 初始种群大小
def population_initialization(G,all_nodes,pop,K):
    P = [ ]
    for _ in range(pop): # ele_9
        P.append(SHD(G,all_nodes,K))
    return P

def Neighbor_Nodeset(G,S):
    neighbors = [ ]
    for i in S:
        neighbors += Neighbor_nodes(G,i)
    neighbors = list(set(neighbors))
    return neighbors

def Neighbor_nodes(G,u):
    return list(G.neighbors(u)) + [u]

def Eval(G,S,K):
    Neighbors = [ ]
    Neighbors = Neighbor_Nodeset(G,S)

    fitness = K
    L = list(set(Neighbors) - set(S))
    for TIME in L:
        fitness += 1 - (1 - 0.01)**len(set(G.neighbors(TIME)) & set(S))
    return fitness

def local_search(G,candidate,G_best,diction,K):

    N_current = G_best
    N_current_copy = N_current.copy()
    for k in range(K):
        flag = False
        Neighbors = list(G.neighbors(N_current_copy[k]))
        while flag == False:
            if len(set(Neighbors)-set(N_current_copy)) != 0:
                for j in random.sample(set(Neighbors)-set(N_current_copy),1):
                    N_current_copy[k] = j
                    if Evaluation(G,diction,N_current_copy,K) > Evaluation(G,diction,N_current,K):
                        N_current = N_current_copy
                    else:
                        flag = True
            else:
                flag = True
        N_current_copy = N_current.copy()
    return N_current_copy

def Evaluation(G,diction,S,K):
    if diction.get(tuple(sorted(S,key=int,reverse=True))) != None:
        evaluation = diction.get(tuple(sorted(S,key=int,reverse=True)))
    else:
        evaluation = Eval(G,S,K)
        diction[tuple(sorted(S,key=int,reverse=True))] = evaluation
    return evaluation

def update_Velocity(Pbest,Gbest,V,X,K,c1,c2,w,r1,r2):
    key_1 = [ ]
    key_2 = [ ]
    V_1 = [ ]
    V_2 = [ ]
    intersaction_1 = set(Pbest) & set(X)
    intersaction_2 = set(Gbest) & set(X)
    for i in range(K):
        if X[i] not in intersaction_1:
            key_1.append(i)
    for _ in range(K): # j
        V_1.append(0)
    for s in key_1:
        V_1[s] = 1

    for n in range(K):
        if X[n] not in intersaction_2:
            key_2.append(n)
    for _ in range(K): # m
        V_2.append(0)
    for e in key_2:
        V_2[e] = 1

    for t in range(K):
        V[t] = w*V[t]+c1*r1*V_1[t] + c2*r2*V_2[t]
    for x in range(K):
        if V[x] < 2:
            V[x] = 0
        else:
            V[x] = 1
    return V

def update_Position(G,all_nodes,Pbest,Gbest,V,X,K,c1,c2,w,r1,r2):
    V = update_Velocity(Pbest,Gbest,V,X,K,c1,c2,w,r1,r2)
    for j in range(K):
        if V[j] == 1:
            for i in random.sample(set(all_nodes)-set(X),1):
                X[j] = i
    return X,V

def main():
    #读取文件，构建网络
    address = "facebook.txt"
    G = nx.read_edgelist(address,create_using=nx.Graph()) ##图G
        ## 粒子群的大小
    maxgen = 100 ##最大迭代次数
    w = 0.8 ##惯性权重
    c1 = 2.0 ##学习因子一
    c2 = 2.0 ##学习因子二
    K = 10 ##种子集大小

    r1 = random.random()
    r2 = random.random()

    all_nodes = list(G.nodes())

    B = [ ]
    C = [ ]
    population_dict = { }            #字典，用来存储个体的适应度值
    P = [ ]
    P_best = [ ]

    #开始时间
    start_time = time.clock()
    pop = 100

    ##种群初始化,位置初始化,每个粒子的位置向量数为 K
    P = population_initialization(G,all_nodes,pop,K)

    #PSO算法的迭代过程
    RESULT_1 = [ ]
    RESULT_2 = [ ]
    P_best = population_initialization(G,all_nodes,pop,K)
    G_best = sorted(P_best,key = lambda x:Evaluation(G,population_dict,x,K),reverse = True)[0]

    ##初始速度,全设置为0,每个粒子的速度向量数为 K
    Velocity = [ ]
    for _ in range(pop): # Index_3
        Velocity.append([ ])
    for Index_1 in range(pop):
        for _ in range(K): # Index_2
            Velocity[Index_1].append(0)
    ## 代数
    g = 1
    while g <= maxgen:
        #更新速度和位置
        for j in range(len(P)):
            P[j],Velocity[j] = update_Position(G,all_nodes,P_best[j],G_best,Velocity[j],P[j],K,c1,c2,w,r1,r2)
        #更新P_best
        for i in range(len(P)):
            if Evaluation(G,population_dict,P[i],K) >= Evaluation(G,population_dict,P_best[i],K):
                P_best[i] = P[i]
        #更新G_best
        G_best_candidate = sorted(P_best,key = lambda x:Evaluation(G,population_dict,x,K),reverse = True)[0]

        G_best_candidate = local_search(G,all_nodes,G_best_candidate,population_dict,K)

        if Evaluation(G,population_dict,G_best,K) < Evaluation(G,population_dict,G_best_candidate,K):
            G_best = G_best_candidate

        diffusion_result = 0
        diffusion_result = Evaluation(G,population_dict,G_best,K)
        RESULT_1.append(g)
        RESULT_2.append(round(diffusion_result,2))

        #print('第',g,'次迭代')
        g += 1
        end_time = time.clock()

        runningtime = end_time - start_time
        C.append(runningtime)

        diffusion_result_list_final = 0
        diffusion_result_list_final = IC_model(G,G_best)
        B.append(diffusion_result_list_final)

        print("网络：",address,"K=",K,"影响力:",round(np.mean(B),1),"时间：",round(np.mean(C),1))
if __name__ == '__main__':
    main()
