import json

data = [0.001, 1.95, 0.002, 1.86, 0.01, 1.63, 0.02, 1.52, 0.04, 1.40, 0.05, 1.36, 0.06, 1.32, 0.08, 1.27, 0.1, 1.22, 0.2, 1.07]

# Преобразование списка в словарь
result_dict = {data[i]: data[i + 1] for i in range(0, len(data), 2)}

# Сохранение словаря в JSON-файл
with open('kolmagorov.json', 'w') as json_file:
    json.dump(result_dict, json_file)

print("Словарь сохранен в файл 'kolmagorov.json'.")
