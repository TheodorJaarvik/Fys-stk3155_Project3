import pandas as pd
import re

class Processing:
    def __init__(self,data):
        self.data = data

    def process(self):
        df = self.data
        filtered_df = df[['text']]
        filtered_df = filtered_df[(filtered_df['text'].notna()) & (filtered_df['text'].str.len() > 0)]
        filtered_df['text'] = filtered_df['text'].str.lower()
        filtered_df['text'] = filtered_df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

        return filtered_df




df1 = pd.read_csv('../data/fake.csv')
df2 = pd.read_csv('../data/true.csv')

df_filtered = Processing(df1).process()
df_filtered['label'] = False
df_filtered_true = Processing(df2).process()
df_filtered_true['label'] = True

df_final = pd.concat([df_filtered, df_filtered_true])
df_final = df_final.sample(frac=1).reset_index(drop=True)

df_final.to_csv('../data/df_final.csv', index=False)


