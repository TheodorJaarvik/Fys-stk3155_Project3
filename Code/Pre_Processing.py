import pandas as pd
import re

class Processing:
    def __init__(self,data):
        self.data = data

    def process(self):
        df = self.data
        filtered_df = df[(df['text'].notna()) & (df['text'].str.len() > 0)]
        filtered_df['text'] = filtered_df['text'].str.lower()
        filtered_df['text'] = filtered_df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

        return filtered_df




