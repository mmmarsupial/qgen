import pandas as pd
from transformers import AutoTokenizer
from prepare_data import DataPrep
from metric import CalculateMetric


model_name = "t5-small"
max_length_in = 200
max_length_out = 100

#tokenizer = AutoTokenizer.from_pretrained(model_name)

#dp = DataPrep(model_name)

#test_data = dp.prepare_data("./claim_datasets/test_short.csv")

#df = pd.DataFrame({
#    'name': ['Alice', 'Bob', 'Charlie'],
#    'age': [25, 30, 35]
#})

metric = CalculateMetric(model_name, metric="rouge")