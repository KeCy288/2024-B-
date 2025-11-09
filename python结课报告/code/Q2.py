#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

plt.rcParams.update({'font.size': 14})
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入表格数据
data = {
    "情况": [1, 2, 3, 4, 5, 6],
    "零配件1_次品率": [10, 20, 10, 20, 10, 5],
    "零配件1_购买单价": [4, 4, 4, 4, 4, 4],
    "零配件1_检测成本": [2, 2, 2, 1, 8, 2],
    "零配件2_次品率": [10, 20, 10, 20, 20, 5],
    "零配件2_购买单价": [18, 18, 18, 18, 18, 18],
    "零配件2_检测成本": [3, 3, 3, 1, 1, 3],
    "成品_次品率": [10, 20, 10, 20, 10, 5],
    "成品_装配成本": [6, 6, 6, 6, 6, 6],
    "成品_检测成本": [3, 3, 3, 2, 2, 3],
    "市场售价": [56, 56, 56, 56, 56, 56],
    "调换损失": [6, 6, 30, 30, 10, 10],
    "拆解费用": [0, 0, 0, 0, 0, 40]
}

df = pd.DataFrame(data)

# 记录前一次企业生产的成品数量
previous_n = 100

def count_profit(opt1, opt2, opt_check, opt_rework, p1, p2, p3, p_check_cost1, 
                p_check_cost2, assembly_cost, p_check_cost, replace_cost, 
                rework_cost, n1, n2):
    """
    计算不同决策组合下的总利润
    """
    global previous_n
    
    # 递归退出条件
    if n1 < 1 or n2 < 1:
        return 0

    # 考虑策略1、2
    p_assembly = 1 - max(opt1 * p1, opt2 * p2)
    product_n = n1 * p_assembly

    # 理想状态下，完全没有次品的成品个数
    if opt1 == 1 and opt2 == 1:
        ideal_n = n1 * (1 - max(p1, p2)) * (1 - p3)
    else:
        ideal_n = n1 * (1 - p1) * (1 - p2) * (1 - p3)

    # 根据策略3，得到实际成品数量
    real_n = ideal_n if opt_check == 1 else product_n

    # 计算隐藏次品率
    if real_n != 0:
        p_hidden = 1 - ideal_n / real_n
    else:
        p_hidden = 0

    # 各项成本计算
    cost_parts = opt1 * p_check_cost1 * n1 + opt2 * p_check_cost2 * n2
    cost_assembly = assembly_cost * product_n
    cost_product = opt_check * p_check_cost * product_n
    cost_rework = opt_rework * rework_cost * (product_n - ideal_n)
    cost_replace = replace_cost * p_hidden * real_n

    # 总成本与总收入
    total_cost = cost_parts + cost_assembly + cost_product + cost_rework + cost_replace
    total_revenue = ideal_n * 56
    
    # 递归利润计算
    reject_n = opt_rework * (product_n - ideal_n)
    is_earned = ideal_n * 34 - total_cost
    
    additional_profit = 0
    if is_earned > 0 and previous_n - 1 > reject_n:
        previous_n = reject_n
        additional_profit = f_profit(opt1, opt2, opt_check, opt_rework, p1, p2, p3,
                                   p_check_cost1, p_check_cost2, assembly_cost,
                                   p_check_cost, replace_cost, rework_cost, 
                                   reject_n, reject_n)

    total_profit = total_revenue - total_cost + additional_profit
    return total_profit

def f_profit(opt1, opt2, opt_check, opt_rework, p1, p2, p3, p_check_cost1, 
             p_check_cost2, assembly_cost, p_check_cost, replace_cost, 
             rework_cost, n1, n2):
    """
    递归函数，先进行次品率的概率修正
    """
    # 次品率修正公式
    if opt1 == 0 and opt2 == 0:
        p1_new = p1 / (1 - (1 - p1) * (1 - p2) * (1 - p3))
        p2_new = p2 / (1 - (1 - p1) * (1 - p2) * (1 - p3))
        p3_new = p3 / (1 - (1 - p1) * (1 - p2) * (1 - p3))
    elif opt1 == 1 and opt2 == 0:
        p1_new = 0
        p2_new = p2 / (1 - (1 - p2) * (1 - p3))
        p3_new = p3 / (1 - (1 - p2) * (1 - p3))
    elif opt1 == 0 and opt2 == 1:
        p1_new = p1 / (1 - (1 - p1) * (1 - p3))
        p2_new = 0
        p3_new = p3 / (1 - (1 - p1) * (1 - p3))
    else:
        p1_new = 0
        p2_new = 0
        p3_new = 1
        
    return count_profit(opt1, opt2, opt_check, opt_rework, p1_new, p2_new, p3_new,
                       p_check_cost1, p_check_cost2, assembly_cost, p_check_cost,
                       replace_cost, rework_cost, n1, n2)

# 遍历所有决策组合
decisions = list(itertools.product([0, 1], repeat=4))
results = []

for index, row in df.iterrows():
    for decision in decisions:
        previous_n = 100
        opt1, opt2, opt_check, opt_rework = decision
        
        # 参数提取和转换
        p1 = row['零配件1_次品率'] / 100
        p2 = row['零配件2_次品率'] / 100
        p3 = row['成品_次品率'] / 100
        assembly_cost = row['成品_装配成本']
        p_check_cost = row['成品_检测成本']
        replace_cost = row['调换损失']
        rework_cost = row['拆解费用']
        p_check_cost1 = row['零配件1_检测成本']
        p_check_cost2 = row['零配件2_检测成本']
        
        # 计算利润
        profit = count_profit(opt1, opt2, opt_check, opt_rework, p1, p2, p3,
                            p_check_cost1, p_check_cost2, assembly_cost,
                            p_check_cost, replace_cost, rework_cost, 100, 100)
        
        results.append({
            "情况": row["情况"],
            "零件1检测": decision[0],
            "零件2检测": decision[1],
            "成品检测": decision[2],
            "不合格成品拆解": decision[3],
            "利润": profit - (4 + 18) * 100  # 减去零件购买成本
        })

# 结果分析
results_df = pd.DataFrame(results)
best_decisions = results_df.loc[results_df.groupby("情况")["利润"].idxmax()]

print("最佳决策方案：")
print(best_decisions)

# 可视化1：各情况最佳决策利润
plt.figure(figsize=(10, 6))
cases = [f'情况{i}' for i in range(1, 7)]
profits = [best_decisions[best_decisions['情况'] == i]['利润'].values[0] for i in range(1, 7)]

bars = plt.bar(cases, profits, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
plt.title('各情况下的最佳决策利润', fontsize=16)
plt.xlabel('情况', fontsize=14)
plt.ylabel('利润', fontsize=14)

# 添加数值标签
for bar, profit in zip(bars, profits):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             f'{profit:.2f}', ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q2_sample_size_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# 可视化2：所有策略利润变化
plt.figure(figsize=(12, 6))
unique_decisions = results_df[['零件1检测', '零件2检测', '成品检测', '不合格成品拆解']].drop_duplicates()

for index, decision in unique_decisions.iterrows():
    strategy_rows = results_df[
        (results_df['零件1检测'] == decision['零件1检测']) &
        (results_df['零件2检测'] == decision['零件2检测']) &
        (results_df['成品检测'] == decision['成品检测']) &
        (results_df['不合格成品拆解'] == decision['不合格成品拆解'])
    ]
    plt.plot(strategy_rows['情况'], strategy_rows['利润'], marker='o', 
             label=f"策略: {decision.values}")

plt.title('6种情况在不同策略下的总利润变化')
plt.xlabel('情况')
plt.ylabel('总利润')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Q2_sample_size_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

