#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#问题四修正后的次品率代码
import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_min_sample_size(p_total, p0=0.1, confidence_level=0.95, error_margin=0.05):
    """计算Cochran公式下的最小样本量"""
    Z = norm.ppf(confidence_level)
    m = (Z ** 2 * p0 * (1 - p0)) / (error_margin ** 2)
    m_adjusted = m / (1 + (m - 1) / p_total)
    return int(np.ceil(m_adjusted))

def adjust_defect_rate(original_rate, p_total, p0=0.1, confidence_level=0.95, error_margin=0.05):
    """修正次品率"""
    sample_size = calculate_min_sample_size(p_total, p0, confidence_level, error_margin)
    Z = norm.ppf(confidence_level)
    margin_of_error = Z * np.sqrt((original_rate * (1 - original_rate)) / sample_size)
    adjusted_rate = original_rate + margin_of_error
    return min(adjusted_rate, 1)

# 问题二数据修正
data_q2 = {
    '情况': [1, 2, 3, 4, 5, 6],
    '零配件1_次品率': [0.1, 0.2, 0.1, 0.2, 0.1, 0.05],
    '零配件2_次品率': [0.1, 0.2, 0.1, 0.2, 0.2, 0.05],
    '成品_次品率': [0.1, 0.2, 0.1, 0.2, 0.1, 0.05]
}

df_q2 = pd.DataFrame(data_q2)
p_total = 10000

# 计算修正后的次品率
adjusted_rates_q2 = []
for _, row in df_q2.iterrows():
    adjusted_rate_1 = adjust_defect_rate(row['零配件1_次品率'], p_total)
    adjusted_rate_2 = adjust_defect_rate(row['零配件2_次品率'], p_total)
    adjusted_rate_finished = adjust_defect_rate(row['成品_次品率'], p_total)
    
    adjusted_rates_q2.append({
        '情况': row['情况'],
        '零配件1_次品率': round(adjusted_rate_1, 4),
        '零配件2_次品率': round(adjusted_rate_2, 4),
        '成品_次品率': round(adjusted_rate_finished, 4)
    })

adjusted_df_q2 = pd.DataFrame(adjusted_rates_q2)
print("问题二修正后的次品率：")
print(adjusted_df_q2)

