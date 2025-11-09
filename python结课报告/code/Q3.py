#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 16})
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class MultiStageProduction:
    def __init__(self):
        # 常量参数
        self.p_check_cost1 = 1  # 零件1、2检测成本
        self.p_check_cost2 = 1  
        self.p_check_cost3 = 2  # 零件3检测成本
        self.assembly_cost = 8  # 装配成本
        self.replace_cost = 40  # 成品调换损失
        self.mid_check_cost = 4  # 半成品检测成本
        self.p_check_cost = 6   # 成品检测成本
        self.mid_rework_cost = 6  # 半成品拆解费用
        self.p_rework_cost = 10   # 成品拆解费用
        
        # 初始次品率
        self.p_part = 0.1  # 零件次品率
        self.p_mid = 0.1   # 半成品次品率
        
        # 递归记录
        self.previous_mid_n = 100
        self.previous_f_n = 100

    def count_profit(self, opt1, opt2, opt3, opt4, opt5, n, p_part, p_mid, depth=0):
        """
        零件-成品全流程利润计算函数
        """
        # 递归深度限制，防止无限递归
        if depth > 10:
            return 0
            
        # 递归退出条件
        if n < 1:
            return 0

        # ==================== 零件 -> 半成品阶段 ====================
        
        # 计算半成品生产数量
        if opt1 == 1:  # 检测零件
            # 检测后只有合格零件进入装配
            p_to_mid = (1 - p_part) * 0.9  # 零件合格率 × 装配合格率
        else:  # 不检测零件
            # 所有零件都进入装配，考虑零件次品率和装配次品率
            p_to_mid = (1 - p_part)**3 * 0.9  # 3个零件都合格的概率 × 装配合格率
            
        product_mid_n = n * p_to_mid

        # 理想半成品数量（完全没有次品）
        if opt1 == 1:
            ideal_mid_n = n * (1 - p_part) * 0.9
        else:
            ideal_mid_n = n * (1 - p_part)**3 * 0.9

        # 实际半成品数量（根据检测决策）
        real_mid_n = ideal_mid_n if opt2 == 1 else product_mid_n

        # 计算隐藏次品率
        if real_mid_n > 0:
            p_mid_hidden = 1 - ideal_mid_n / real_mid_n
        else:
            p_mid_hidden = 0

        # 半成品阶段成本计算
        cost_parts = 0
        if opt1 == 1:
            # 检测所有8种零件
            cost_parts = (6 * n * self.p_check_cost1 +  # 6个成本为1的零件
                         2 * n * self.p_check_cost3)   # 2个成本为2的零件

        cost_mid_assembly = self.assembly_cost * 3 * product_mid_n  # 装配成本
        cost_mid_check = opt2 * self.mid_check_cost * 3 * product_mid_n  # 检测成本
        cost_mid_rework = opt4 * self.mid_rework_cost * 3 * max(0, product_mid_n - ideal_mid_n)  # 拆解成本

        # 半成品拆解递归
        reject_mid_n = opt4 * max(0, product_mid_n - ideal_mid_n)
        additional_profit1 = 0

        if reject_mid_n > 0 and self.previous_mid_n > reject_mid_n:
            self.previous_mid_n = reject_mid_n
            # 修正次品率
            if opt1 == 0 and opt2 == 0:
                p_part_new = p_part / (1 - (1 - p_part)**3 * 0.9)
            else:
                p_part_new = p_part
                
            additional_profit1 = self.count_profit(opt1, opt2, opt3, opt4, opt5, 
                                                 reject_mid_n, p_part_new, p_mid, depth+1)

        # ==================== 半成品 -> 成品阶段 ====================
        
        # 计算成品生产数量
        if opt2 == 1:  # 检测半成品
            p_to_f = (1 - p_mid) * 0.9  # 半成品合格率 × 装配合格率
        else:  # 不检测半成品
            p_to_f = (1 - p_mid)**3 * 0.9  # 3个半成品都合格的概率 × 装配合格率
            
        product_f_n = product_mid_n * p_to_f

        # 理想成品数量
        if opt2 == 1:
            ideal_f_n = product_mid_n * (1 - p_mid) * 0.9
        else:
            ideal_f_n = product_mid_n * (1 - p_mid)**3 * 0.9

        # 实际成品数量
        real_f_n = ideal_f_n if opt3 == 1 else product_f_n

        # 计算隐藏次品率
        if real_f_n > 0:
            p_f_hidden = 1 - ideal_f_n / real_f_n
        else:
            p_f_hidden = 0

        # 成品阶段成本计算
        cost_f_assembly = self.assembly_cost * product_f_n
        cost_f_product = opt3 * self.p_check_cost * product_f_n
        cost_f_rework = opt5 * self.p_rework_cost * max(0, product_f_n - ideal_f_n)
        cost_replace = self.replace_cost * p_f_hidden * real_f_n

        # 成品拆解递归
        reject_f_n = opt5 * max(0, product_f_n - ideal_f_n)
        additional_profit2 = 0

        if reject_f_n > 0 and self.previous_f_n > reject_f_n:
            self.previous_f_n = reject_f_n
            
            if opt4 == 1 and opt5 == 1:
                # 都拆回零件
                if opt1 == 0 and opt2 == 0:
                    p_part_new = p_part / (1 - (1 - p_part)**3 * 0.9)
                else:
                    p_part_new = p_part
                additional_profit2 = self.count_profit(opt1, opt2, opt3, opt4, opt5,
                                                     reject_f_n, p_part_new, p_mid, depth+1)
                    
            elif opt4 == 0 and opt5 == 1:
                # 只拆解成品，得到半成品
                if opt2 == 0 and opt3 == 0:
                    p_mid_new = p_mid / (1 - (1 - p_mid)**3 * 0.9)
                else:
                    p_mid_new = p_mid
                additional_profit2 = self.count_middle_to_final(opt2, opt3, reject_f_n, p_mid_new, depth+1)

        # ==================== 总利润计算 ====================
        
        total_cost = (cost_parts + cost_mid_assembly + cost_mid_check + 
                     cost_mid_rework + cost_f_assembly + cost_f_product + 
                     cost_f_rework + cost_replace)
        
        total_revenue = ideal_f_n * 200  # 市场售价200元
        
        total_profit = total_revenue - total_cost + additional_profit1 + additional_profit2

        return total_profit

    def count_middle_to_final(self, opt2, opt3, mid_n, p_mid, depth=0):
        """
        半成品->成品阶段利润计算（当只拆解成品时）
        """
        if depth > 10 or mid_n < 1:
            return 0

        # 成品生产计算
        if opt2 == 1:  # 检测半成品
            p_to_f = (1 - p_mid) * 0.9
        else:  # 不检测半成品
            p_to_f = (1 - p_mid)**3 * 0.9
            
        product_n = mid_n * p_to_f

        # 理想成品数量
        if opt2 == 1:
            ideal_f_n = mid_n * (1 - p_mid) * 0.9
        else:
            ideal_f_n = mid_n * (1 - p_mid)**3 * 0.9

        # 实际成品数量
        real_f_n = ideal_f_n if opt3 == 1 else product_n

        # 隐藏次品率
        if real_f_n > 0:
            p_f_hidden = 1 - ideal_f_n / real_f_n
        else:
            p_f_hidden = 0

        # 成本计算
        cost_mid_check = opt2 * self.mid_check_cost * 3 * mid_n
        cost_f_assembly = self.assembly_cost * product_n
        cost_f_product = opt3 * self.p_check_cost * product_n
        cost_f_rework = 1 * self.p_rework_cost * max(0, product_n - ideal_f_n)  # opt5=1
        cost_replace = self.replace_cost * p_f_hidden * real_f_n

        # 递归处理
        reject_f_n = max(0, product_n - ideal_f_n)
        additional_profit = 0

        if reject_f_n > 0 and self.previous_f_n > reject_f_n:
            self.previous_f_n = reject_f_n
            if opt2 == 0 and opt3 == 0:
                p_mid_new = p_mid / (1 - (1 - p_mid)**3 * 0.9)
            else:
                p_mid_new = p_mid
            additional_profit = self.count_middle_to_final(opt2, opt3, reject_f_n, p_mid_new, depth+1)

        total_cost = cost_mid_check + cost_f_assembly + cost_f_product + cost_f_rework + cost_replace
        total_revenue = ideal_f_n * 200
        total_profit = total_revenue - total_cost + additional_profit

        return total_profit

def main():
    # 创建生产模型实例
    production = MultiStageProduction()
    
    # 定义所有决策组合（32种）
    decisions = list(itertools.product([0, 1], repeat=5))
    results = []
    
    print("开始计算32种策略的利润...")
    
    for i, decision in enumerate(decisions):
        opt1, opt2, opt3, opt4, opt5 = decision
        
        # 重置递归记录
        production.previous_mid_n = 100
        production.previous_f_n = 100
        
        try:
            # 计算利润（减去零件购买成本）
            profit = production.count_profit(opt1, opt2, opt3, opt4, opt5, 100, 0.1, 0.1)
            net_profit = profit - 64 * 100  # 减去零件总成本
            
            results.append({
                "策略编号": i + 1,
                "零件检测": opt1,
                "半成品检测": opt2,
                "成品检测": opt3,
                "不合格半成品拆解": opt4,
                "不合格成品拆解": opt5,
                "利润": net_profit
            })
            
            if (i + 1) % 8 == 0:
                print(f"已完成 {i + 1}/32 种策略计算")
                
        except Exception as e:
            print(f"策略 {i+1} 计算错误: {e}")
            results.append({
                "策略编号": i + 1,
                "零件检测": opt1,
                "半成品检测": opt2,
                "成品检测": opt3,
                "不合格半成品拆解": opt4,
                "不合格成品拆解": opt5,
                "利润": -10000  # 错误标记
            })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 找到最优策略
    best_strategy_idx = results_df['利润'].idxmax()
    best_strategy = results_df.loc[best_strategy_idx]
    
    print(f"\n{'='*50}")
    print("最优策略分析结果")
    print(f"{'='*50}")
    print(f"最优策略编号: {best_strategy['策略编号']}")
    print(f"决策组合: ({best_strategy['零件检测']}, {best_strategy['半成品检测']}, {best_strategy['成品检测']}, "
          f"{best_strategy['不合格半成品拆解']}, {best_strategy['不合格成品拆解']})")
    print(f"最大利润: {best_strategy['利润']:.2f} 元")
    
    # 策略描述
    strategy_desc = {
        '零件检测': '检测' if best_strategy['零件检测'] == 1 else '不检测',
        '半成品检测': '检测' if best_strategy['半成品检测'] == 1 else '不检测',
        '成品检测': '检测' if best_strategy['成品检测'] == 1 else '不检测',
        '半成品拆解': '拆解' if best_strategy['不合格半成品拆解'] == 1 else '不拆解',
        '成品拆解': '拆解' if best_strategy['不合格成品拆解'] == 1 else '不拆解'
    }
    
    print("\n最优策略描述:")
    for key, value in strategy_desc.items():
        print(f"  {key}: {value}")
    
    # 可视化结果
    visualize_results(results_df, best_strategy)

def visualize_results(results_df, best_strategy):
    """可视化分析结果"""
    
    # 1. 所有策略利润对比图
    plt.figure(figsize=(15, 8))
    
    # 主图：所有策略利润
    plt.subplot(2, 1, 1)
    colors = ['red' if x == best_strategy['策略编号'] - 1 else 
             'lightcoral' if results_df.loc[x, '利润'] < 0 else 
             'steelblue' for x in range(len(results_df))]
    
    bars = plt.bar(results_df['策略编号'], results_df['利润'], color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=best_strategy['利润'], color='red', linestyle='--', 
                label=f'最大利润: {best_strategy["利润"]:.2f}')
    
    plt.title('32种生产策略利润对比分析', fontsize=16, fontweight='bold')
    plt.xlabel('策略编号', fontsize=12)
    plt.ylabel('利润 (元)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 最优策略详细分析
    plt.subplot(2, 1, 2)
    
    # 提取前10个最优策略
    top_10 = results_df.nlargest(10, '利润')
    
    plt.bar(range(len(top_10)), top_10['利润'], color='lightgreen', alpha=0.7)
    plt.axhline(y=best_strategy['利润'], color='red', linestyle='--', 
                label=f'最优策略 {best_strategy["策略编号"]}')
    
    plt.title('前10个最优策略利润对比', fontsize=14)
    plt.xlabel('排名')
    plt.ylabel('利润 (元)')
    plt.xticks(range(len(top_10)), [f'第{idx+1}名\n策略{row["策略编号"]}' 
                                   for idx, (_, row) in enumerate(top_10.iterrows())], 
               rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Q3_sample_size_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 策略分布统计
    print(f"\n{'='*50}")
    print("策略分布统计分析")
    print(f"{'='*50}")
    
    positive_profits = results_df[results_df['利润'] > 0]
    negative_profits = results_df[results_df['利润'] < 0]
    zero_profits = results_df[results_df['利润'] == 0]
    
    print(f"盈利策略数量: {len(positive_profits)} ({len(positive_profits)/32*100:.1f}%)")
    print(f"亏损策略数量: {len(negative_profits)} ({len(negative_profits)/32*100:.1f}%)")
    print(f"零利润策略数量: {len(zero_profits)} ({len(zero_profits)/32*100:.1f}%)")
    print(f"平均利润: {results_df['利润'].mean():.2f} 元")
    print(f"利润标准差: {results_df['利润'].std():.2f} 元")

if __name__ == "__main__":
    main()

