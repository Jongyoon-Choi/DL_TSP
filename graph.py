import matplotlib.pyplot as plt

# 데이터
alpha_values = [0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.00001]
num_simulations_values = [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 1000000, 3000000, 5000000, 10000000]
total_cost_values = [61.901, 61.055, 59.506, 59.246, 59.172, 59.172, 59.325, 59.345, 61.457, 65.1275, 70.499, 25.318, 24.444, 30.526]

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(num_simulations_values, total_cost_values, marker='o', linestyle='-')

# 그래프 제목 및 레이블 설정
plt.title('Visualization')
plt.xlabel('Number of Simulations')
plt.ylabel('Total Cost')
plt.xscale('log')  # x축 로그 스케일로 설정

# 각 데이터 포인트 옆에 alpha 값을 텍스트로 표시
#for i, alpha in enumerate(alpha_values):
#    plt.text(num_simulations_values[i], total_cost_values[i], f'alpha={alpha}', fontsize=9, verticalalignment='bottom', horizontalalignment='right')

# 그래프 표시
plt.grid(True)
plt.show() 