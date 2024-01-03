# External imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline

# Local imports
from metric import CalculateMetric
from prepare_data import DataPrep

# Variables
model_name = "./checkpoints/t5-small-fine-tuned/checkpoint-1000"
use_full_data = True

if use_full_data:
    paths = {"train": "./claim_datasets/train.csv",
            "validation": "./claim_datasets/valid.csv",
            "test": "./claim_datasets/test.csv"
            }
else:
    paths = {"train": "./claim_datasets/train_short.csv",
            "validation": "./claim_datasets/valid_short.csv",
            "test": "./claim_datasets/test_short.csv"
            }

# Prepare test data
data_prep = DataPrep(model_name=model_name)
test_data = data_prep.prepare_data(paths["test"])

# Pipeline for text generation
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = Text2TextGenerationPipeline(model, tokenizer)
metric = CalculateMetric(model_name=model_name, metric="rouge")

# Generates questions for each claim in the test dataset and calculates the score for each generated question
def generate_questions(dataset):
    claims = dataset["claim_reviewed"]#.to_list()
    gold_questions = dataset["question"]#.to_list()

    for claim, gold_question in zip(claims, gold_questions):
        generated_question = pipeline(claim,
                                      do_sample=True, 
                                      num_return_sequences=1, 
                                      num_beams=3, 
                                      top_k=200,
                                      top_p=1,
                                      temperature=1, 
                                      #max_new_tokens=20, 
                                      #min_new_tokens=8, 
                                      )                                                                             # Pipeline outputs a list of dicts (in this case one dict). "generated_text" is the key for the output text given a claim
        scores = metric.compute_metrics_detokenized(([generated_question],[gold_question]))
        print("Input: " + claim + "\n")
        print("Generated output: " + generated_question[0]["generated_text"] + "\n")                               # generated_question = [{"generated_text": "This is the pipeline output"}]
        print("Gold output: " + gold_question + "\n")
        print(scores)
        print("~"*50)

generate_questions(test_data)
        





