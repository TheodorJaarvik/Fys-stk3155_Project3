from Pre_Processing import Processing
import pandas as pd


df = pd.read_csv('../data/fake.csv')
df2 = pd.read_csv('../data/true.csv')

df_filtered = Processing(df).process()
df_filtered_true = Processing(df2).process()

print(df.shape, df2.shape)
print(df_filtered.shape, df_filtered_true.shape)
df_filtered.head()
