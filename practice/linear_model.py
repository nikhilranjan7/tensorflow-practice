import pandas as pd
import tempfile

# Reding and labeling dataset
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
train_file.name = "./data/train.csv"
test_file.name = "./data/test.csv"

CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"]

df_train = pd.read_csv(train_file.name, names=CSV_COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file.name, names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)
