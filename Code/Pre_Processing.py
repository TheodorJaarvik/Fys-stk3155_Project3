import pandas as pd

# Import true and fake data
fake_path = 'data/fake.csv'
fake_data = pd.read_csv(fake_path)

true_path = 'data/true.csv'
true_data = pd.read_csv(true_path)

# Check for missing values in the 'text' column of the true and fake dataset
fake_text_missing = fake_data['text'].isna().any()
fake_text_missing_count = fake_text_missing.sum()

true_text_missing = true_data['text'].isna().any()
true_text_missing_count = true_text_missing.sum()

print(fake_text_missing, fake_text_missing_count)
print(true_text_missing, true_text_missing_count)

