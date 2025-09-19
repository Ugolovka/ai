import pandas as pd
from sklearn.preprocessing import OneHotEncoder



data = pd.DataFrame({"fruit": ["apple", "orange", "banan", "pineapple", "orange", "apple"],
                     "weight":[100,80,50,90,125,150]
})

print(data)
encoder = OneHotEncoder()
encoded = encoder.fit_transform(data[["fruit"]])

print(encoded.toarray())

print(encoder.get_feature_names_out(["fruit"]))