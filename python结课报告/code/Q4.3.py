#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#问题四用遗传算法重新计算第三题
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 问题三数据结构
data = {
    '零配件': {
        1: {'次品率': 0.15, '购买单价': 2, '检测成本': 1},
        2: {'次品率': 0.15, '购买单价': 8, '检测成本': 1},
        3: {'次品率': 0.15, '购买单价': 12, '检测成本': 2},
        4: {'次品率': 0.15, '购买单价': 2, '检测成本': 1},
        5: {'次品率': 0.15, '购买单价': 8, '检测成本': 1},
        6: {'次品率': 0.15, '购买单价': 12, '检测成本': 2},
        7: {'次品率': 0.15, '购买单价': 8, '检测成本': 1},
        8: {'次品率': 0.15, '购买单价': 12, '检测成本': 2},
    },
    '半成品': {
        1: {'次品率': 0.15, '装配成本': 8, '检测成本': 4, '拆解成零件': {1: 2, 2: 1}},
        2: {'次品率': 0.15, '装配成本': 8, '检测成本': 4, '拆解成零件': {3: 1, 4: 1}},
        3: {'次品率': 0.15, '装配成本': 8, '检测成本': 4, '拆解成零件': {5: 2, 6: 1}},
    },
    '成品': {
        '次品率': 0.15,
        '装配成本': 8,
        '拆解费用': 10,
        '拆解成半成品': {1: 2, 2: 1}
    },
    '市场售价': 200,
    '调换损失': 40
}

# 遗传算法参数
population_size = 100
num_generations = 200
mutation_rate = 0.15
tournament_size = 5
elite_size = 10

def initialize_population(pop_size, num_genes):
    return np.random.randint(2, size=(pop_size, num_genes))

def calculate_fitness(individual, data):
    """计算多工序生产适应度"""
    total_cost, total_revenue, undetected_parts_loss = 0, 0, 0

    # 零配件成本计算
    for i in range(8):
        inspect = individual[i]
        if i + 1 in data['零配件']:
            part_data = data['零配件'][i + 1]
            cost = (part_data['次品率'] * part_data['购买单价'] + part_data['检测成本']) * inspect
            total_cost += cost
            if inspect == 0:
                undetected_parts_loss += part_data['检测成本']

    # 半成品成本计算
    for j in range(3):
        inspect = sum(individual[8 + 3 * j: 11 + 3 * j])  # 半成品相关零件检测
        if j + 1 in data['半成品']:
            half_product = data['半成品'][j + 1]
            cost = (half_product['次品率'] * half_product['装配成本'] + 
                   half_product['检测成本']) * inspect
            total_cost += cost

            # 拆解零件减少成本
            for part_id, amount in half_product['拆解成零件'].items():
                if part_id in data['零配件']:
                    part_cost = (data['零配件'][part_id]['次品率'] * 
                               data['零配件'][part_id]['购买单价'] * amount)
                    total_cost -= part_cost
                    total_revenue += data['市场售价'] - data['调换损失']
                    if individual[part_id - 1] == 0:  # 未检测零配件的拆解损失
                        undetected_parts_loss += data['零配件'][part_id]['检测成本']

    # 成品成本计算
    product_inspect = individual[17]
    cost = (data['成品']['次品率'] * data['成品']['装配成本']) * product_inspect
    total_cost += cost
    total_revenue += data['市场售价'] - data['调换损失']

    # 成品拆解
    if individual[18] == 1:
        total_cost += data['成品']['拆解费用']
        for semi_id, amount in data['成品']['拆解成半成品'].items():
            if semi_id in data['半成品']:
                semi_cost = (data['半成品'][semi_id]['次品率'] * 
                           data['半成品'][semi_id]['装配成本']) * amount
                total_cost -= semi_cost
                total_revenue += data['市场售价'] - data['调换损失']

                for part_id, part_amount in data['半成品'][semi_id]['拆解成零件'].items():
                    if part_id in data['零配件']:
                        part_cost = (data['零配件'][part_id]['次品率'] * 
                                   data['零配件'][part_id]['购买单价']) * part_amount
                        total_cost -= part_cost
                        total_revenue += data['市场售价'] - data['调换损失']

    # 总成本考虑未检测零配件的损失
    total_cost += undetected_parts_loss
    return total_revenue - total_cost

def select(population, fitness, tournament_size):
    """锦标赛选择"""
    if tournament_size > len(population):
        tournament_size = len(population)
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    selected_fitness = fitness[selected_indices]
    best_index = selected_indices[np.argmax(selected_fitness)]
    return population[best_index]

def crossover(parent1, parent2):
    """单点交叉"""
    point = np.random.randint(1, len(parent1))
    return (np.concatenate((parent1[:point], parent2[point:])),
            np.concatenate((parent2[:point], parent1[point:])))

def mutate(individual, mutation_rate):
    """变异操作"""
    mutation_mask = np.random.rand(len(individual)) < mutation_rate
    individual[mutation_mask] = 1 - individual[mutation_mask]
    return individual

def genetic_algorithm(population_size, num_genes, num_generations, mutation_rate, 
                     elite_size, tournament_size, data):
    """遗传算法主函数"""
    population = initialize_population(population_size, num_genes)
    best_fitness_over_time = []
    avg_fitness_over_time = []

    for generation in range(num_generations):
        fitness = np.array([calculate_fitness(ind, data) for ind in population])
        best_fitness_over_time.append(np.max(fitness))
        avg_fitness_over_time.append(np.mean(fitness))
        
        # 精英选择
        elite = np.array([select(population, fitness, tournament_size) 
                         for _ in range(elite_size)])
        new_population = elite.copy()

        # 交叉变异
        while len(new_population) < population_size:
            parent1 = select(population, fitness, tournament_size)
            parent2 = select(population, fitness, tournament_size)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population = np.vstack((new_population, child1, child2))

        population = new_population[:population_size]

    best_fitness = np.max(fitness)
    best_individual = population[np.argmax(fitness)]
    return best_individual, best_fitness, best_fitness_over_time, avg_fitness_over_time

# 运行遗传算法
num_genes = 19
print("开始问题三遗传算法优化...")
best_individual, best_fitness, best_fitness_over_time, avg_fitness_over_time = genetic_algorithm(
    population_size, num_genes, num_generations, mutation_rate, elite_size, tournament_size, data
)

print(f"最佳适应度: {best_fitness:.2f}")

# 翻译为决策表格
def translate_to_decision(best_individual, data):
    decisions = {
        '零配件': {i + 1: '检测' if best_individual[i] == 1 else '不检测' for i in range(8)},
        '半成品': {j + 1: '检测' if sum(best_individual[8 + 3 * j: 11 + 3 * j]) > 0 else '不检测' for j in range(3)},
        '成品检测': '检测' if best_individual[17] == 1 else '不检测',
        '成品拆解': '拆解' if best_individual[18] == 1 else '不拆解'
    }
    decisions['利润'] = best_fitness
    return decisions

decisions = translate_to_decision(best_individual, data)
print("最佳决策方案：")
for key, value in decisions.items():
    print(f"{key}: {value}")

# 可视化进化过程
plt.figure(figsize=(12, 6))
plt.plot(best_fitness_over_time, 'b-', linewidth=2, label='最佳适应度')
plt.plot(avg_fitness_over_time, 'r--', linewidth=1, label='平均适应度', alpha=0.7)
plt.xlabel('迭代代数')
plt.ylabel('适应度')
plt.title('遗传算法适应度进化过程（问题三）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Q5_sample_size_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存决策结果
df_decision = pd.DataFrame([decisions])
df_decision.to_excel('Q4_best_individual_decision.xlsx', index=False)

