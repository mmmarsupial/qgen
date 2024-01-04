import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset



class DataPrep:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.prefix = "claim: "
        self.suffix = "\nquestions related to claim:"
        self.max_length_in = 200
        self.max_length_out = 100

    def prepare_data(self, path):

        df = pd.read_csv(path, 
                        usecols=["claim_reviewed", "question"],
                        encoding="utf-8").dropna()                                                      # Reads only claim and question column from file. Drops rows with None-values
        
        prompted_claims = df["claim_reviewed"].apply(self._add_prefix_suffix)                           # Applies prefix and suffix to every claim to create prompt
        df["claim_reviewed"] = prompted_claims                                                          # Replaces the claim column with the prompt column

        tokenized_claims = self.tokenizer(df["claim_reviewed"].to_list(),                               # Tokenizes the claim prompts
                                          max_length=self.max_length_in, 
                                          truncation=True)
        tokenzied_questions = self.tokenizer(df["question"].to_list(),                                  # Tokenizes the questions
                                             max_length=self.max_length_out, 
                                             truncation=True)

        df.insert(0, "input_ids", tokenized_claims["input_ids"])                                        # Extracts the claim prompts from tokenizer output. Adds to dataframe
        df.insert(0, "attention_mask", tokenized_claims["attention_mask"])                              # Extracts the attention mask from tokenizer output. Adds to dataframe
        df.insert(0, "labels", tokenzied_questions["input_ids"])                                        # Extracts the questions from tokenizer output. Adds to dataframe

        return Dataset.from_pandas(df)                                                                  # Returns as a dataset (DataSet class) instead of pandas dataframe
        
    def _add_prefix_suffix(self, claim):
        prompt = self.prefix + claim + self.suffix                                                      # Used in prepare data to att the pre-/suffix to claim.
        return prompt
        

        

    
    