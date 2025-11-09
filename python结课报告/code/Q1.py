#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

plt.rcParams.update({'font.size': 16})
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_min_sample_size(p, p0=0.1, confidence_level=0.95, error_margin=0.05):
    """
    计算给定总体数量 p 下的最小样本量 m，使得在95%置信水平下能够判断是否拒收。
    """
    Z = norm.ppf(confidence_level)  # 单侧检验，正态分布查表得到Z值
    # 使用 Cochran 公式计算样本量
    m = (Z ** 2 * p0 * (1 - p0)) / (error_margin ** 2)

    # 如果总体数量有限，使用有限总体修正公式
    if p < np.inf:
        m_adjusted = m / (1 + (m - 1) / p)
    else:
        m_adjusted = m

    return int(np.ceil(m_adjusted))  # 向上取整以确保足够的样本量

# 1. 95%置信水平 - 小样本
p_values_small = np.arange(1, 502, 20)
m_values_95_small = [calculate_min_sample_size(p) for p in p_values_small]

# 2. 95%置信水平 - 大样本
p_values_large = np.arange(500, 100001, 2000)
m_values_95_large = [calculate_min_sample_size(p) for p in p_values_large]

# 3. 90%置信水平 - 小样本
m_values_90_small = [calculate_min_sample_size(p, confidence_level=0.9) for p in p_values_small]

# 4. 90%置信水平 - 大样本
m_values_90_large = [calculate_min_sample_size(p, confidence_level=0.9) for p in p_values_large]

# 绘制综合图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 95%置信水平小样本
ax1.plot(p_values_small, m_values_95_small, 'bo-', linewidth=2, markersize=4)
ax1.set_title('95%置信水平 - 小样本')
ax1.set_xlabel('总体数量 p')
ax1.set_ylabel('最小样本量 m')
ax1.grid(True)

# 95%置信水平大样本
ax2.plot(p_values_large, m_values_95_large, 'bo-', linewidth=2, markersize=4)
ax2.set_title('95%置信水平 - 大样本')
ax2.set_xlabel('总体数量 p')
ax2.set_ylabel('最小样本量 m')
ax2.grid(True)

# 90%置信水平小样本
ax3.plot(p_values_small, m_values_90_small, 'ro-', linewidth=2, markersize=4)
ax3.set_title('90%置信水平 - 小样本')
ax3.set_xlabel('总体数量 p')
ax3.set_ylabel('最小样本量 m')
ax3.grid(True)

# 90%置信水平大样本
ax4.plot(p_values_large, m_values_90_large, 'ro-', linewidth=2, markersize=4)
ax4.set_title('90%置信水平 - 大样本')
ax4.set_xlabel('总体数量 p')
ax4.set_ylabel('最小样本量 m')
ax4.grid(True)

plt.tight_layout()
plt.savefig('Q1_sample_size_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出关键结果
print("95%置信水平下，当总体数量足够大时，最小样本量趋于:", calculate_min_sample_size(100000))
print("90%置信水平下，当总体数量足够大时，最小样本量趋于:", calculate_min_sample_size(100000, confidence_level=0.9))

