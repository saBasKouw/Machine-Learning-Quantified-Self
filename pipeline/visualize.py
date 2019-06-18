import pandas as pd

df = pd.read_csv("chapter2_result.csv")
df.index = pd.to_datetime(df.index)

#Missing values
print(len(df))

#Outliers
