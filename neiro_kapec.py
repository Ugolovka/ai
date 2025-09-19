from sklearn.model_selection import train_test_split  # делим данные на train/test
from sklearn.tree import DecisionTreeClassifier       # модель: дерево решений
from sklearn.metrics import accuracy_score, classification_report  # метрики
import numpy as np

# категории: 0 = дешёво, 1 = дорого
X = np.array([[25],[23],[26],[4],[2],[18],[1],[19],[3],[4],[2],[14],[11],[1],[12],[2],[7],[1],[6],[5],[7],[22],[35],[14],[3],[28],[19],[11],[39],[26]])  # цены
y = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1])  # категории
names = ["смартфон", "планшет", "ноутбук", "чашка", "стикеры", "наушники", "карандаш",
    "клавиатура", "ластик", "блокнот", "флешка", "книга", "ручка", "маркер",
    "батарейка", "салфетки", "мышка", "скрепка", "коврик", "зарядка",
    "флешка", "книга", "ручка", "маркер", "батарейка", "салфетки",
    "мышка", "скрепка", "коврик", "зарядка"]

items = [{"название": name, "цена": price[0], "категория": "дорогой" if label == 1 else "дешёвый"}
         for name, price, label in zip(names, X, y)]

cheap_items = [item for item in items if item["категория"] == "дешёвый"]
expensive_items = [item for item in items if item["категория"] == "дорогой"]

print("Дешёвые товары:")
for item in cheap_items:
    print(f'{item["название"]}: {item["цена"]} EURO')

print("============"
    "Дорогие товары:")
for item in expensive_items:
    print(f'{item["название"]}: {item["цена"]} EURO')

# Делим данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Создаём дерево решений (глубина = 3)
clf = DecisionTreeClassifier(max_depth=3, random_state=1)

# Обучаем дерево
clf.fit(X_train, y_train)

# Делаем предсказания
pred = clf.predict(X_test)

# Считаем точность и выводим отчёт
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=["дешёвый", "дорогой"]))

# Проверка на новой цене
sample = [[4]]  # цена товара
print("Предсказанная категория:", ["дешёвый", "дорогой"][clf.predict(sample)[0]])