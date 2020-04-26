import networkx as nx
import numpy as np
import random
from networkx.algorithms import community

# 标签传播算法
# 输入:图G
# 输出:社区划分结果(划分结果用子图节点二维数组表示)
def label_propagation_community(G):
    communities_generator = list(community.label_propagation_communities(G))
    m = []
    for i in communities_generator:
        m.append(list(i))
    return m


#------#




# Degree_Based_Initialization(G,n,k);
# input G,n,k
# output X

## x中保存由种子集定义的粒子位置向量
def SHD(G,all_nodes,k):
    x = [ ]
    x = all_nodes[:k] # all_nodes已根据节点度来排过序了
    for i in range(k):
        if random.random() > 0.5:
            for ii in random.sample(set(all_nodes)-set(x),1):
                x[i] = ii
    return x
## 种群初始化,粒子位置初始化,每个粒子的位置向量数为k,共有n个粒子
def population_initialization(G,all_nodes,n,k):
    P = [ ]
    for _ in range(n):
        P.append(SHD(G,all_nodes,k))
    return P


def EDV(G,X,k):
    Neighbors = []
    for i in X:
        Neighbors += list(G.neighbors(i))
    Neighbors += X
    Neighbors = list(set(Neighbors))
    fitness = k
    L = list(set(Neighbors) - set(X))
    for TIME in L:
        fitness += 1 - (1 - 0.01)**len(set(G.neighbors(TIME)) & set(X))
    return fitness

# def Evaluation(G, Pbest_s_fitness, X, k):
#     if diction.get(tuple(sorted(S,key=int,reverse=True))) != None:
#         evaluation = diction.get(tuple(sorted(S,key=int,reverse=True)))
#     else:
#         evaluation = Eval(G,X,k)
#         diction[tuple(sorted(S,key=int,reverse=True))] = evaluation
#     return evaluation


# local_search(G, all_nodes, Gbest_candidate, Gbest_fitness, k)
def local_search(G, Gbest_candidate, Gbest_fitness, k):
    N_current = Gbest_candidate
    N_current_copy = N_current.copy()
    for i in range(k):
        flag = False
        Neighbors = list(G.neighbors(N_current_copy[i]))
        while flag == False:
            if len(set(Neighbors)-set(N_current_copy)) != 0:
                for ii in random.sample(set(Neighbors)-set(N_current_copy),1):
                    N_current_copy[i] = ii
                    N_current_copy_fitness = EDV(G,N_current_copy,k)
                    if N_current_copy_fitness > Gbest_fitness:
                        N_current = N_current_copy
                    else:
                        flag = True
            else:
                flag = True
        N_current_copy = N_current.copy()
    return N_current_copy



def update_Velocity(Pbest, Gbest, V, X, k, c1, c2, w):
    r1 = random.random()
    r2 = random.random()
    key_1 = [ ]
    key_2 = [ ]
    V_1 = [ ]
    V_2 = [ ]
    intersaction_1 = set(Pbest) & set(X)
    intersaction_2 = set(Gbest) & set(X)
    for i in range(k):
        if X[i] not in intersaction_1:
            key_1.append(i)
    for _ in range(k):
        V_1.append(0)
    for i in key_1:
        V_1[i] = 1

    for i in range(k):
        if X[i] not in intersaction_2:
            key_2.append(i)
    for _ in range(k): # m
        V_2.append(0)
    for i in key_2:
        V_2[i] = 1

    for i in range(k):
        V[i] = w*V[i]+c1*r1*V_1[i] + c2*r2*V_2[i]
    for i in range(k):
        if V[i] < 2:
            V[i] = 0
        else:
            V[i] = 1
    return V

def update_Position(G,all_nodes,Pbest,Gbest,V,X,k,c1,c2,w):
    V = update_Velocity(Pbest,Gbest,V,X,k,c1,c2,w)
    for i in range(k):
        if V[i] == 1:
            for ii in random.sample(set(all_nodes)-set(X),1):
                X[i] = ii
    return X,V




def IC_model(g,X,mc=10000,p=0.01):
    spread = []
    for _ in range(mc):
        # 模拟传播过程
        new_active, Au = X[:], X[:]
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







