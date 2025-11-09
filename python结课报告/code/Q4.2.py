#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#问题四用遗传算法重新计算第二题
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 使用修正后的数据
data = {
    '情况': [1, 2, 3, 4, 5, 6],
    '零配件1_次品率': [0.15, 0.27, 0.15, 0.27, 0.15, 0.09],
    '零配件2_次品率': [0.15, 0.27, 0.15, 0.27, 0.27, 0.09],
    '零配件1_购买单价': [4, 4, 4, 4, 4, 4],
    '零配件1_检测成本': [2, 2, 2, 1, 8, 2],
    '零配件2_购买单价': [18, 18, 18, 18, 18, 18],
    '零配件2_检测成本': [3, 3, 3, 1, 1, 3],
    '成品_次品率': [0.15, 0.27, 0.15, 0.27, 0.15, 0.09],
    '装配成本': [6, 6, 6, 6, 6, 6],
    '市场售价': [56, 56, 56, 56, 56, 56],
    '调换损失': [6, 6, 30, 30, 10, 10],
    '拆解费用': [5, 5, 5, 5, 5, 40]
}

# 遗传算法参数
population_size = 50
num_generations = 100
mutation_rate = 0.1
tournament_size = 5
elite_size = 5

def initialize_population(pop_size, num_genes):
    """初始化种群"""
    return np.random.randint(2, size=(pop_size, num_genes))

def calculate_fitness(individual, data):
    """计算适应度（利润）"""
    total_cost = 0
    total_revenue = 0

    for i in range(len(data['情况'])):
        part1_inspect = individual[i * 4]      # 零配件1检测
        part2_inspect = individual[i * 4 + 1]  # 零配件2检测
        product_inspect = individual[i * 4 + 2] # 成品检测
        discard_defective = individual[i * 4 + 3] # 不合格成品拆解

        if discard_defective == 1:  # 拆解的情况
            total_cost += data['拆解费用'][i]
            if part1_inspect == 1:
                total_cost += (data['零配件1_次品率'][i] * data['零配件1_购买单价'][i] + 
                             data['零配件1_检测成本'][i])
                total_revenue += data['市场售价'][i] - data['调换损失'][i]
            if part2_inspect == 1:
                total_cost += (data['零配件2_次品率'][i] * data['零配件2_购买单价'][i] + 
                             data['零配件2_检测成本'][i])
                total_revenue += data['市场售价'][i] - data['调换损失'][i]
        else:  # 不拆解的情况
            if part1_inspect == 1:
                total_cost += (data['零配件1_次品率'][i] * data['零配件1_购买单价'][i] + 
                             data['零配件1_检测成本'][i])
            if part2_inspect == 1:
                total_cost += (data['零配件2_次品率'][i] * data['零配件2_购买单价'][i] + 
                             data['零配件2_检测成本'][i])
            if product_inspect == 1:
                total_cost += data['成品_次品率'][i] * data['装配成本'][i]
                total_revenue += data['市场售价'][i] - data['调换损失'][i]

    return total_revenue - total_cost

def select(population, fitness):
    """锦标赛选择"""
    selected_indices = np.argsort(fitness)[-tournament_size:]
    return population[selected_indices]

def crossover(parent1, parent2):
    """单点交叉"""
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def mutate(individual, mutation_rate):
    """变异操作"""
    mutation_mask = np.random.rand(len(individual)) < mutation_rate
    individual[mutation_mask] = 1 - individual[mutation_mask]
    return individual

def local_search(individual):
    """局部搜索优化"""
    best = individual.copy()
    best_fitness = calculate_fitness(best, data)
    
    for i in range(len(best)):
        new_individual = best.copy()
        new_individual[i] = 1 - new_individual[i]
        new_fitness = calculate_fitness(new_individual, data)
        if new_fitness > best_fitness:
            best, best_fitness = new_individual, new_fitness
            
    return best

def genetic_algorithm(data):
    """遗传算法主程序"""
    num_genes = len(data['情况']) * 4
    population = initialize_population(population_size, num_genes)
    fitness_history = []

    for generation in range(num_generations):
        # 计算适应度
        fitness = np.array([calculate_fitness(ind, data) for ind in population])
        selected = select(population, fitness)

        # 生成新种群
        new_population = []
        while len(new_population) < population_size - elite_size:
            parent1, parent2 = selected[np.random.choice(len(selected), 2, replace=False)]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        # 精英保留
        elite_indices = np.argsort(fitness)[-elite_size:]
        elite = population[elite_indices]
        new_population.extend(elite)
        population = np.array(new_population)

        # 记录适应度历史
        fitness_history.append(np.max(fitness))

        # 增加多样性
        num_replacements = int(population_size * 0.1)
        population[:num_replacements] = initialize_population(num_replacements, num_genes)

    # 最终优化
    final_fitness = np.array([calculate_fitness(ind, data) for ind in population])
    best_index = np.argmax(final_fitness)
    best_individual = population[best_index]
    best_individual = local_search(best_individual)
    best_fitness = calculate_fitness(best_individual, data)

    return best_individual, best_fitness, fitness_history

# 运行遗传算法
print("开始遗传算法优化...")
best_decision, best_score, fitness_history = genetic_algorithm(data)

print(f"最佳适应度值: {best_score:.2f}")

# 解析最优解
results = []
for i, situation in enumerate(data['情况']):
    decision = best_decision[i * 4:(i + 1) * 4]
    results.append([
        situation,
        decision[0],  # 零配件1检测
        decision[1],  # 零配件2检测
        decision[2],  # 成品检测
        decision[3],  # 拆解
    ])

# 输出结果
df_result = pd.DataFrame(results, columns=['情况', '零配件1检测', '零配件2检测', '成品检测', '拆解'])
print("最佳决策方案：")
print(df_result)

# 可视化进化过程
plt.figure(figsize=(10, 6))
plt.plot(range(len(fitness_history)), fitness_history, 'b-', linewidth=2)
plt.xlabel('迭代代数')
plt.ylabel('最佳适应度')
plt.title('遗传算法适应度进化过程（问题二）')
plt.grid(True, alpha=0.3)
plt.savefig('Q4_sample_size_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

