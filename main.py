import math
from prettytable import PrettyTable

import matplotlib.pyplot as plt

row = [11.94, 12.38, 9.00, 12.06, 9.32,
       11.88, 8.07, 12.69, 13.59, 10.36,
       13.27, 10.04, 8.30, 8.61, 9.04,
       8.65, 11.99, 8.99, 9.16, 12.69,
       9.30, 11.88, 12.02, 11.69, 12.76,
       9.11, 12.76, 9.68, 8.69, 12.30,
       8.10, 9.50, 9.44, 12.32, 9.77,
       13.63, 9.64, 9.88, 12.21, 13.40,
       9.56, 11.11, 10.23, 9.49, 8.86,
       11.13, 9.98, 8.56, 8.43, 7.87]

# Количество элементов выборки
n = len(row)

# Максимальное и минимальное значения выборки
max_value = max(row)
min_value = min(row)

# Размах выборки (R)
R = round(max_value - min_value, 2)

# Шаг выборки
step = round(R / (1 + math.log2(n)), 1)

# Выводим основную информацию
print(f"Количество элементов выборки: {n}")
print(f"Максимальное значение выборки: {max_value}")
print(f"Минимальное значение выборки: {min_value}")
print(f"Размах выборки (R): {R}")
print(f"Шаг выборки: {step}")
print(f"Отсортированный список: {sorted(row)}")

# Определение интервалов
a0 = min_value - step / 2
intervals = []

for i in range(1, 8):
    left_bound = a0 + step * (i - 1)
    right_bound = left_bound + step
    intervals.append((round(left_bound, 2), round(right_bound, 2)))

# Среднее значение и количество значений в каждом интервале
table = PrettyTable()
table.field_names = ["Интервал", "Среднее значение", "Штрих-столбец", "Частота", "Относительная частота",
                     "Плотность частоты"]

for interval in intervals:
    left_bound, right_bound = interval
    values_in_interval = [value for value in row if left_bound <= value <= right_bound]
    len_values_in_interval = len(values_in_interval)

    mean_value = sum(values_in_interval) / len(values_in_interval) if len(values_in_interval) > 0 else None
    mean_value = round(mean_value, 2) if mean_value else None

    relative_frequency = len(values_in_interval) / n
    relative_frequency = round(relative_frequency, 2)

    density = relative_frequency / step
    density = round(density, 2)

    table.add_row([f"{left_bound}-{right_bound}", mean_value, len_values_in_interval, len_values_in_interval,
                   relative_frequency, density])

# Вывод таблицы
print(table)

# Построение гистограммы относительных частот
plt.figure(figsize=(10, 6))  # Задаем размер графика
plt.bar([f"{left}-{right}" for left, right in intervals], [row[4] for row in table._rows], width=0.8)  # Изменяем ширину столбцов
plt.xlabel('Интервалы')
plt.ylabel('Относительная частота')
plt.title('Гистограмма относительных частот')

# Сохраняем график в файл
plt.savefig('histogram.png')

plt.show()