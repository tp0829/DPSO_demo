import networkx as nx
from networkx.algorithms import community

# 标签传播算法
# 输入:图G
# 输出:社区划分结果(划分结果用二维列表表示)
def label_propagation_community(G):
    communities_generator = list(community.label_propagation_communities(G))
    m = []
    for i in communities_generator:
        m.append(list(i))
    return m

G_1 = nx.read_edgelist("twitter.txt", create_using=nx.Graph())  # 读取文件，构建网络
G_2 = nx.read_edgelist("facebook.txt", create_using=nx.Graph())  # 读取文件，构建网络
G_3 = nx.read_edgelist("hamster.txt", create_using=nx.Graph())  # 读取文件，构建网络

g1 = label_propagation_community(G_1)
g2 = label_propagation_community(G_2)
g3 = label_propagation_community(G_3)

n1 = []
n2 = []
n3 = []

for i in g1:
    n1.append(len(i))
for i in g2:
    n2.append(len(i))
for i in g3:
    n3.append(len(i))

a1 = sorted(n1, reverse=True)
a2 = sorted(n2, reverse=True)
a3 = sorted(n3, reverse=True)

'''
networkx.classes.function.subgraph(G, nbunch)
'''
