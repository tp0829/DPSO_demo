import numpy as np
import matplotlib.pyplot as plt

'''
np.random.uniform(-1,1,(10,2)):生成10*2的数组,数组元素大小[-1,1)



'''

class DPSO(object):
    def __init__(self, population_size, seed_size, max_steps):
        self.n = population_size  # 粒子群大小
        self.gmax = max_steps  # 最大迭代次数
        self.w = 0.6  # 惯性权重
        self.c1 = self.c2 = 2
        self.k=seed_size    #种子集大小

        # 初始化粒子群位置        
        self.dim = 2  # 搜索空间的维度
        self.x_bound = [-10, 10]  # 解空间范围
        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],
                                   (self.n, self.dim))  
        self.v = np.random.rand(self.n, self.dim)  # 初始化粒子群速度
        fitness = self.calculate_fitness(self.x)
        self.p = self.x  # 个体的最佳位置
        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度

    def calculate_fitness(self, x):
        return np.sum(np.square(x), axis=1)

    def evolve(self):
        plt.figure()
        for _ in range(self.gmax):
            '''np.random.rand(10,2):生成10*2的数组'''
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
