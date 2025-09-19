from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

text=[
    "Перевод 200 руб. на карту ****1234 ", # банк
    "Приходи в гости вечером",# не банк
    "Оплата на сумму 560 руб. прошла успешно",# банк
    "Перевод 228 руб. на карту ****2284 не успешно", # банк
    "Приходи на хату вечером",# не банк
    "Оплата по счёту 987 руб. проведена успешно",# банк
    "Скинь 1 руб. на мою карту ****1234 ", # не банк
    "Приходи ко мне вечером",# не банк
    "Перевод на счёт компании 33000 руб. выполнен успешно",# банк
]

lable=[1,0,1,1,0,1,0,0,1] # 1 - банк, 0 - не банк

text_train, text_test, y_train, y_test = train_test_split(text, lable, test_size=0.33, random_state=42)

print(text_train)
print(text_test)
print(y_train)
print(y_test)
pipe = make_pipeline(
    CountVectorizer(), #задача сделать мешок слов, это подсчёт ко-лва вхождения слов игнорируя их порядковый номер, и на выводе получается матрица
    MultinomialNB()
)
pipe.fit(text_train, y_train)
y_pred = pipe.predict(text_test)
print(f"Accuracy: {accuracy_score(y_test,y_pred)}")

print("New: ", pipe.predict(["Перевод 200 руб. на карту ****1234"])[0])
print("New: ", pipe.predict(["Приходи ко мне вечером"])[0])