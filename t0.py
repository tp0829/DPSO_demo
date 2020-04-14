# 笔记 2020年04月14日

import networkx as nx
import matplotlib.pyplot as plt

G_1 = nx.read_edgelist("twitter.txt",create_using=nx.Graph())  #读取文件，构建网络 
G_2 = nx.read_edgelist("facebook.txt",create_using=nx.Graph())  #读取文件，构建网络 
G_3 = nx.read_edgelist("hamster.txt",create_using=nx.Graph())  #读取文件，构建网络    

len(G_1.nodes()) 
len(G_1.edges())