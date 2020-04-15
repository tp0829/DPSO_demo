# 粒子群优化算法,PSO:Particle Swarm Optimization
# 学习笔记
'''
0.0 参考
    - 最优化算法之粒子群算法（PSO）
        https://blog.csdn.net/daaikuaichuan/article/details/81382794
1.0 算法分析
    - 基本思想:通过 群体中个体之间的协作和信息共享 来寻找最优解.
    - 优势:
        1. 实现简单
        2. 没有许多参数要调节
    - 更新规则:
        公式一: X_i = X_i + V_i
        公式二: V_i = W*V_i 
                + C1*rand()*(pbest_i-X_i) 
                + C2*rand()*(gbest_i-X_i)
            // W:惯性因子,其值为负
                其值较大,全局寻优能力强,局部寻优能力弱
                其值较小,全局寻优能力弱,局部寻优能力强
            // rand():介于(0,1)之间的随机数
            // C1,C2:学习因子,通常C1=C2=2
            // pbest:自身最好的一个矢量
            // gbest:全局最好的一个矢量
2.0 算法流程
    - 1)开始
    - 2)随机初始化每个粒子
    - 3)评估每个粒子并得到全局最优
    - 4)判断是否满足结束条件
        (转结束or继续)
    - 5)更新每个粒子的速度和位置
    - 6)评估每个粒子的函数适应值,判断当前位置和原位置比哪个更好
    - 7)更新每个粒子的历史最优位置
    - 8)更新群体的全局最优位置
        (转步骤4)
    - 9)结束
3.0 算法实现,以Python为例相关库函数
    - import numpy as np
        np.random.uniform(-1,1,(10,2)):生成10*2的数组,数组元素大小[-1,1)
        np.random.rand(10,2):生成10*2的数组,数组元素大小随机默认范围[0,1)
        np.sum(x,axis=1):axis=0,往里看0层,=1,往里看1层
        np.argmin(x):返回数组x中最小元素的下标
    - import matplotlib.pyplot as plt
        plt.figure():
        plt.show():
        plt.clf():清除所有轴,窗口打开,重复利用窗口的,已达到动画显示的效果

'''

import numpy as np
import matplotlib.pyplot as plt


class DPSO(object):
    # 初始化每个粒子
    def __init__(self, population_size, seed_size, max_steps):
        self.n = population_size  # 粒子群大小
        self.gmax = max_steps  # 最大迭代次数
        self.w = 0.6  # 惯性权重
        self.c1 = self.c2 = 2
        self.k=seed_size    # 种子集大小 ## 没有使用到
        self.dim = 2  # 搜索空间的维度
        self.x_bound = [-10, 10]  # 解空间范围
        ''' 生成n*dim的数组,数组元素大小[-10,10) '''
        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],
                                   (self.n, self.dim))  # 初始化粒子群的位置
        ''' 生成n*dim的数组,数组元素随机大小[0,1) '''
        self.v = np.random.rand(self.n, self.dim)  # 初始化粒子群速度
        fitness = self.calculate_fitness(self.x)
        self.p = self.x  # 个体的最佳位置
        '''np.argmin(x):返回数组x中最小元素的下标'''
        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度

    # 计算适应度
    def calculate_fitness(self, x):
        '''np.sum(x,axis=1):axis=0,往里看0层,=1,往里看1层'''
        return np.sum(np.square(x), axis=1)

    def evolve(self):
        plt.figure()
        for _ in range(self.gmax):
            ''' 生成n*dim的数组,数组元素随机大小[0,1) '''
            r1 = np.random.rand(self.n, self.dim)
            r2 = np.random.rand(self.n, self.dim)
            # 更新速度和权重
            self.v = self.w*self.v+self.c1*r1 * \
                (self.p-self.x)+self.c2*r2*(self.pg-self.x)
            self.x = self.v + self.x
            plt.clf()
            plt.scatter(self.x[:, 0], self.x[:, 1], s=30, color='k')
            plt.xlim(self.x_bound[0], self.x_bound[1])
            plt.ylim(self.x_bound[0], self.x_bound[1])
            plt.pause(0.01)
            fitness = self.calculate_fitness(self.x)
            # 需要更新的个体
            '''np.greater(2,3):返回2>3的布尔值'''
            update_id = np.greater(self.individual_best_fitness, fitness)
            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)
            print('best fitness: %.5f, mean fitness: %.5f' %
                  (self.global_best_fitness, np.mean(fitness)))


pso = DPSO(100,10,100)
pso.evolve()
plt.show()
