import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('2024_AI_TSP.csv', header=None)
x_coords = data.iloc[:, 0].values
y_coords = data.iloc[:, 1].values

solution = pd.read_csv('Part1_solution.csv', header=None)
solution = solution.iloc[:, 0].values

plt.figure(figsize=(10, 6))
for i in range(len(solution)-1):
    plt.plot([x_coords[solution[i]], x_coords[solution[i+1]]], [y_coords[solution[i]], y_coords[solution[i+1]]], color='blue')


plt.plot([x_coords[solution[-1]], x_coords[solution[0]]], [y_coords[solution[-1]], y_coords[solution[0]]], color='blue')

plt.scatter(x_coords, y_coords, color='black', marker='o')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Path')


plt.grid(True)
plt.show()