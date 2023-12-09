import json
import math

import matplotlib.pyplot as plt
import scipy
from prettytable import PrettyTable
from scipy.stats import norm, chi2

# Вариант 4
# Задаем отсортированный список
row = [11.94, 12.38, 9.00, 12.06, 9.32, 11.88, 8.07, 12.69, 13.59, 10.36, 13.27, 10.04, 8.30, 8.61, 9.04, 8.65, 11.99,
       8.99, 9.16, 12.69, 9.30, 11.88, 12.02, 11.69, 12.76, 9.11, 12.76, 9.68, 8.69, 12.30, 8.10, 9.50, 9.44, 12.32,
       9.77, 13.63, 9.64, 9.88, 12.21, 13.40, 9.56, 11.11, 10.23, 9.49, 8.86, 11.13, 9.98, 8.56, 8.43, 7.87]


# Задаем значение β (доверительная вероятность)
beta = 0.92

# Уровень значимости
alpha = 0.08

# Вариант 18
row =[
    10.68, 12.21, 11.70, 10.57, 12.71,
    12.64, 11.90, 10.42, 11.51, 12.58,
    11.21, 11.51, 11.72, 12.50, 11.73,
    12.81, 13.18, 11.35, 11.89, 11.32,
    11.65, 10.85, 10.88, 12.90, 10.39,
    10.78, 12.87, 10.41, 12.04, 10.40,
    11.28, 12.50, 11.95, 11.82, 11.51,
    11.96, 12.30, 11.30, 11.68, 11.01,
    13.03, 13.58, 12.49, 11.47, 10.59,
    12.07, 12.15, 11.57, 10.87, 11.18
]

beta = 0.9
alpha = 0.02


# Количество элементов выборки
n = len(row)

# Максимальное и минимальное значения выборки
max_value = max(row)
min_value = min(row)

# Размах выборки (R)
R = round(max_value - min_value, 2)

# Оптимальное число интервалов
l = 1 + math.log2(n)

# Шаг выборки
step = round(R / l, 2)

# Вывод общей информации
general_info_table = PrettyTable()
general_info_table.field_names = ["Параметр", "Значение"]
general_info_table.add_row(["Количество элементов выборки", n])
general_info_table.add_row(["Максимальное значение выборки", max_value])
general_info_table.add_row(["Минимальное значение выборки", min_value])
general_info_table.add_row(["Размах выборки (R)", R])
general_info_table.add_row(["Шаг выборки", step])

print("Часть 1. Первичная обработка выборки:")
print(general_info_table)

# Вывод отсортированного списка
# print("Отсортированный список:", sorted(row))

# Определение интервалов
a0 = min_value
intervals = []

lines = math.ceil(l)

for i in range(1, lines + 1):
    left_bound = a0 + step * (i - 1)
    right_bound = left_bound + step
    intervals.append((round(left_bound, 2), round(right_bound, 2)))

# Среднее значение и количество значений в каждом интервале
table = PrettyTable()
table.field_names = ["№", "Интервал", "Среднее значение", "Штрих-столбец", "Частота", "Относительная частота",
                     "Эмп. функция распределения", "Плотность частоты"]

empirical_distribution_function = 0

for i, interval in enumerate(intervals):
    left_bound, right_bound = interval
    values_in_interval = [value for value in row if left_bound <= value < right_bound]
    len_values_in_interval = len(values_in_interval)

    mean_value = sum(values_in_interval) / len(values_in_interval) if len(values_in_interval) > 0 else None
    mean_value = round(mean_value, 2) if mean_value else None

    relative_frequency = len(values_in_interval) / n
    relative_frequency = round(relative_frequency, 2)

    density = relative_frequency / step
    density = round(density, 2)

    empirical_distribution_function += relative_frequency

    table.add_row([i + 1, f"{left_bound}-{right_bound}", mean_value, len_values_in_interval, len_values_in_interval,
                   relative_frequency, round(empirical_distribution_function, 2), density])

# Вывод таблицы
print("\nТаблица 1. Статистический интервальный ряд распределения")
print(table)

relative_frequencies = [row[5] for row in table._rows]
empirical_distribution = [row[6] for row in table._rows]
mean_values = [row[2] for row in table._rows]


def plot_histogram_and_polygon(intervals, relative_frequencies):
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar([f"{left}-{right}" for left, right in intervals], relative_frequencies, width=1,
                  edgecolor='black', hatch='////', color='white')

    # Добавляем жирные линии
    ax.plot([f"{left}-{right}" for left, right in intervals], relative_frequencies, color='black', linewidth=3)

    ax.set_xlabel('Интервалы')
    ax.set_ylabel('Относительная частота')
    ax.set_title('Полигон и гистограмма относительных частот')

    # Добавляем точки в середину каждого столбца
    for i, bar in enumerate(bars):
        x_mid = (bar.get_x() + bar.get_width() / 2)
        y_mid = bar.get_height()  # Выберите подходящую высоту для точек
        plt.scatter(x_mid, y_mid, color='black', s=40, zorder=5)

        # Добавляем математическую надпись перед сноской
        math_label = fr"$f_{{{i + 1}}}^*(x)={y_mid:.2f}$"
        plt.annotate(math_label, (x_mid, y_mid), textcoords="offset points", xytext=(0, 5), ha='center', va='bottom',
                     fontsize=11, color='black')

        # Добавляем сноску на верхней части столбца
        plt.annotate(f"{y_mid:.2f}", (x_mid, y_mid / 2), textcoords="offset points", xytext=(0, -10), ha='center',
                     va='bottom',
                     fontsize=15, color='black', fontweight='bold')

    # Увеличиваем ось OY сверху
    ax.tick_params(axis='y', pad=15)

    # Устанавливаем верхнее значение для y
    plt.ylim(0, max(relative_frequencies) + 0.1)

    # Сохраняем график в файл
    plt.savefig('histogram.png')

    plt.show()


plot_histogram_and_polygon(intervals, relative_frequencies)

# Вычисление выборочного среднего
sample_mean = round(sum(rel_freq * mean_value for rel_freq, mean_value in zip(relative_frequencies, mean_values)), 2)

# Вычисление квадрата выборочного среднего
sample_mean_squared = sum(
    (mean_value ** 2 * rel_freq) for rel_freq, mean_value in zip(relative_frequencies, mean_values))

# Вычисление выборочной дисперсии
sample_variance = round(sample_mean_squared - sample_mean ** 2, 2)

# Вычисление выборочного среднего квадратического отклонения
sample_standard_deviation = round(math.sqrt(sample_variance), 2)

# Исправленная выборочная дисперсия
corrected_sample_variance = round((n * sample_variance) / (n - 1), 2)

# Исправленное выборочное среднее квадратическое отклонение
corrected_sample_standard_deviation = round(math.sqrt(corrected_sample_variance), 2)

# Вычисление выборочной моды
if n % 2 == 0:
    # Если n четное, берем элементы с индексами n/2 и n/2 + 1
    mode_index_1 = (n // 2)
    mode_index_2 = (n // 2) + 1
    sample_median = (sorted(row)[mode_index_1 - 1] + sorted(row)[mode_index_2 - 1]) / 2
else:
    # Если n нечетное, берем элемент с индексом (n - 1)/2 + 1
    mode_index = ((n - 1) // 2) + 1
    sample_median = sorted(row)[mode_index - 1]

# Вычисление выборочной моды
mode_counts = {}
for value in row:
    if value in mode_counts:
        mode_counts[value] += 1
    else:
        mode_counts[value] = 1

# Находим значение с наибольшей частотой
max_mode_count = max(mode_counts.values())
sample_mode = [value for value, count in mode_counts.items() if count == max_mode_count]

# Новая таблица для данных о выборочном среднем, выборочной дисперсии и выборочном среднем квадратическом отклонении
corrected_mean_and_variance_table = PrettyTable()
corrected_mean_and_variance_table.field_names = ["Параметр", "Значение"]
corrected_mean_and_variance_table.add_row(["Выборочное среднее X", sample_mean])
corrected_mean_and_variance_table.add_row(["Выборочная дисперсия D", sample_variance])
corrected_mean_and_variance_table.add_row(["Выборочное среднее квадратическое отклонение σ", sample_standard_deviation])
corrected_mean_and_variance_table.add_row(["Исправленная выборочная дисперсия S^2", corrected_sample_variance])
corrected_mean_and_variance_table.add_row(
    ["Исправленное выборочное среднее квадратическое отклонение S", corrected_sample_standard_deviation])
corrected_mean_and_variance_table.add_row(["Выборочная медиана Me", sample_median])
corrected_mean_and_variance_table.add_row(["Выборочная мода Mo", ", ".join(map(str, sample_mode))])

print("\nЧасть 2. Точечные и интервальные оценки параметров распределения:")
print("\nТочечные оценки числовых характеристик:")
print(corrected_mean_and_variance_table)

# Вычисляем значение коэффициента Стьюдента
t_value = round(scipy.stats.t.ppf((1 + beta) / 2, n - 1), 2)

# Вычисляем величину доверительного интервала (точность)
epsilon = round(t_value * sample_standard_deviation / math.sqrt(n), 4)

# Вычисляем границы доверительного интервала
confidence_interval_lower = round(sample_mean - epsilon, 2)
confidence_interval_upper = round(sample_mean + epsilon, 2)

# Создаем таблицу для данных о доверительном интервале
confidence_interval_table = PrettyTable()
confidence_interval_table.field_names = ["Параметр", "Значение"]
confidence_interval_table.add_row(["Доверительная вероятность (β)", beta])
confidence_interval_table.add_row(["Коэффициент Стьюдента", t_value])
confidence_interval_table.add_row(["Точность (эпсилон)", epsilon])

# Выводим таблицу с информацией о доверительном интервале
print("\nИнтервальные оценки числовых характеристик:")

print("Доверительный интервал для математического ожидания в предположении, что дисперсия известна:")
print(confidence_interval_table)
print(f"Доверительный интервал: ({sample_mean} - {epsilon}; {sample_mean} + {epsilon})")


# Величина доверительного интервала для неизвестной дисперсии
epsilon_unknown_variance = round((t_value * corrected_sample_standard_deviation / math.sqrt(n)), 4)

confidence_interval_unknown_variance_table = PrettyTable()
confidence_interval_unknown_variance_table.field_names = ["Параметр", "Значение"]
confidence_interval_unknown_variance_table.add_row(["Уровень значимости (альфа)", alpha])
confidence_interval_unknown_variance_table.add_row(["Коэффициент Стьюдента", t_value])
confidence_interval_unknown_variance_table.add_row(["Точность (эпсилон)", epsilon_unknown_variance])
# Выводим таблицу с информацией о доверительном интервале (дисперсия неизвестна)

print("\nДоверительный интервал для математического ожидания в предположении, что дисперсия неизвестна")
print(confidence_interval_unknown_variance_table)
print(
    f"Доверительный интервал: ({sample_mean} - {epsilon_unknown_variance}; {sample_mean} + {epsilon_unknown_variance})")

print("\nЧасть 3 Статистическая проверка статистических гипотез")

# Получаем столбцы с номером интервала, интервалами, частотой, средним значением и исправленным стандартным отклонением
interval_numbers = [row[0] for row in table._rows]
frequencies = [row[4] for row in table._rows]

# Создаем новую таблицу для проверки статистических гипотез с помощью критерия Пирсона
hypothesis_testing_table = PrettyTable()
hypothesis_testing_table.field_names = ["Номер интервала", "Интервал", "Частота", "Значение X1", "Значение X2",
                                        "Функция Лапласа X1", "Функция Лапласа X2", "Вероятность попадания в интервал",
                                        "nT", "Хи-квадрат"]

# Загрузка значений функции Лапласа из файла laplas.json
with open('laplas.json', 'r') as file:
    laplas_data = json.load(file)


def calculate_laplace_cdf(value):
    rounded_value = round(value, 2)
    rounded_value_str = "{:.2f}".format(rounded_value)
    return laplas_data.get(rounded_value_str, None)


# Добавляем скопированные столбцы в новую таблицу
for number, (left, right), frequency in zip(interval_numbers, intervals, frequencies):
    standardized_X1 = (left - sample_mean) / corrected_sample_standard_deviation
    standardized_X2 = (right - sample_mean) / corrected_sample_standard_deviation


    if standardized_X1 < 0:
        laplace_X1 = -calculate_laplace_cdf(-standardized_X1)
    else:
        laplace_X1 = calculate_laplace_cdf(standardized_X1)

    if standardized_X2 < 0:
        laplace_X2 = -calculate_laplace_cdf(-standardized_X2)
    else:
        laplace_X2 = calculate_laplace_cdf(standardized_X2)

    print(standardized_X1, standardized_X2, laplace_X1, laplace_X2)

    probability_in_interval = laplace_X2 - laplace_X1
    nT = n * probability_in_interval
    chi_square = ((frequency - nT) ** 2) / nT

    hypothesis_testing_table.add_row(
        [number, f"{left}-{right}", frequency, round(standardized_X1, 2), round(standardized_X2, 2),
         round(laplace_X1, 4), round(laplace_X2, 4), round(probability_in_interval, 4), round(nT, 2),
         round(chi_square, 4)])

print("Таблица 2. Проверка статистических гипотез с помощью критерия Пирсона")
print(hypothesis_testing_table)

# Рассчитываем статистику критерия (Хи-квадрат_набл)
chi_square_observed = sum([row[9] for row in hypothesis_testing_table._rows])

# Рассчитываем число степеней свободы k
degrees_of_freedom = lines - 2 - 1

# Рассчитываем критерий хи-квадрат при m = lines и альфа = alpha
chi_square_critical = chi2.ppf(1 - alpha, degrees_of_freedom)

# Создаем таблицу
results_table = PrettyTable()
results_table.field_names = ["Параметр", "Значение"]
results_table.add_row(["Значение статистики критерия (Хи-квадрат_набл)", f"{chi_square_observed:.4f}"])
results_table.add_row(["Уровень значимости (α)", f"{alpha}"])
results_table.add_row(["Число степеней свободы (k)", f"{degrees_of_freedom}"])
results_table.add_row([f"Критерий хи-квадрат при m = {degrees_of_freedom} и α = {alpha}", f"{chi_square_critical:.4f}"])

# Выводим таблицу
print("\nОбщие сведения для части 3 Статистическая проверка статистических гипотез")
print(results_table)

# Проверяем гипотезу
if chi_square_observed > chi_square_critical:
    print("\nГипотеза отвергается: Есть статистически значимые различия в данных.")
else:
    print("\nГипотеза не отвергается: Статистически значимых различий в данных нет.")
