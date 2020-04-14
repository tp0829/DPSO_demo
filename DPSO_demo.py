# 离散粒子群优化算法,DPSO
# Python代码

#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import networkx as nx
import time
import numpy as np
import random
import matplotlib.pyplot as plt

def IC_model(g,S,mc=10000,p=0.01):
    spread = []
    for i in range(mc):
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
      
def SHD(G,all_nodes,K):
    X = [ ] 
    
    candidate = sorted(all_nodes,key=G.nodes,reverse=True)
    X = candidate[:K]
    for ele_1 in range(K):
        if random.random() > 0.5:
            for ele_2 in random.sample(set(all_nodes)-set(X),1):
                X[ele_1] = ele_2
    
    #X = random.sample(all_nodes,K)
    return X

def population_initialization(G,all_nodes,pop,K):
    P = [ ]
    for ele_9 in range(pop):
        P.append(SHD(G,all_nodes,K))
    return P
'''
def Eval(G,S,K):
    one_hop = [ ]
    two_hop = [ ]
    one_influence = 0
    two_influence = 0
    fitness = 0
    
    for i_1 in S:
        one_hop += G.neighbors(i_1)
    one_hop = list(set(one_hop)-set(S))
    
    for i_2 in one_hop:
        one_influence += 1 - (1 - 0.01)**len(set(G.neighbors(i_2))&set(S))
    
    for j_1 in one_hop:
        two_hop += G.neighbors(j_1)
    two_hop = list(set(two_hop) - set(S))
    
    for j_2 in two_hop:
        two_influence += 0.01 * len(set(G.neighbors(j_2))&set(one_hop))  
    fitness = K + one_influence + (one_influence*two_influence)/len(one_hop)
    
    return fitness
'''   
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
    for j in range(K):
        V_1.append(0)
    for s in key_1:
        V_1[s] = 1
    
    for n in range(K):
        if X[n] not in intersaction_2:
            key_2.append(n)
    for m in range(K):
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
    for address in ["D:\\GrQc.txt","D:\\HepTh.txt","D:\\HepPh.txt","D:\\epinions.txt","D:\\Email-Enron.txt","D:\\gemsec-RO.txt"]:
        G = nx.read_edgelist(address,create_using=nx.Graph())  #读取文件，构建网络
        for K in [5,10,15,20,25,30]:
            B = [ ]
            C = [ ]
            for A in range(10):
                start_time = time.clock()
                #G = nx.read_edgelist("D:\\Hamsterster full.txt", create_using=nx.Graph())  #读取文件，构建网络 
                #K = 20
                pop = 100
                maxgen = 100
                c1 = 2.0
                c2 = 2.0
                w = 0.8
                r1 = random.random()
                r2 = random.random()
                
                all_nodes = list(G.nodes())
                
                population_dict = { }            #字典，用来存储个体的适应度值
                P = [ ]
                P_best = [ ]    
                P = population_initialization(G,all_nodes,pop,K)  #种群初始化
                #PSO算法的迭代过程
                RESULT_1 = [ ]
                RESULT_2 = [ ]
                P_best = population_initialization(G,all_nodes,pop,K)    
                G_best = sorted(P_best,key = lambda x:Evaluation(G,population_dict,x,K),reverse = True)[0]
                #初始速度
                Velocity = [ ]
                for Index_3 in range(pop):
                    Velocity.append([ ])
                for Index_1 in range(pop):
                    for Index_2 in range(K):
                        Velocity[Index_1].append(0)
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
                '''
                plt.figure(num='Figure 12', figsize=(6,6), dpi=75, facecolor='#FFFFFF', edgecolor='#FF0000')
                plt.xlabel('The Number of Generations')   # x轴标签
                plt.ylabel('The Value of EDV')  # y轴标签
                plt.plot(RESULT_1,RESULT_2)
                plt.legend(loc=4)
                plt.show()
                
                print("总的激活节点数量:",round(diffusion_result_list_final,2)) 
                print("所用时间：",round(runningtime,2))
                print(RESULT_1)
                print(RESULT_2)
                '''
            print("网络：",address,"K=",K,"影响力:",round(np.mean(B),1),"时间：",round(np.mean(C),1))
if __name__ == '__main__':
    main()