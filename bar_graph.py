import matplotlib.pyplot as plt
import numpy as np

alpha_values = [0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.00001]
total_cost_values = [61.901, 61.055, 59.506, 59.246, 59.172, 59.172, 59.325, 59.345, 61.457, 65.1275, 70.499]

plt.figure(figsize=(12, 6))
plt.bar(alpha_values, total_cost_values, width=0.007, align='center')

plt.title('visualization - num_simulations 100000')
plt.xlabel('Alpha Values')
plt.ylabel('Total Cost')
 # y축 범위 설정
plt.ylim(52, 75) 

# y축 눈금 설정
plt.yticks(np.arange(52, 75, 0.7))  
# x축 눈금 설정
plt.xticks(np.arange(0, 0.25, 0.02))  

plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.show()