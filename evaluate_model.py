# External imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline

# Local imports
from metric import CalculateMetric
from prepare_data import DataPrep

# Variables
model_name = "./checkpoints/t5-small-fine-tuned/checkpoint-1000"
#model_name = "./checkpoints/facebook/bart-base-fine-tuned/checkpoint-2200"
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
    claims = dataset["claim_reviewed"]
    gold_questions = dataset["question"]
    n_claims = len(claims)
    path_to_out = "./evaluation_runs/eval-t5-small.txt"
    log_file = open(path_to_out, "w")
    accumulated_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0, "rougeLsum": 0}


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
        
        
        for key in accumulated_scores:
            accumulated_scores[key] += scores[key]


        log_file.write("Input: " + claim + "\n")
        log_file.write("Generated output: " + generated_question[0]["generated_text"] + "\n")                               # generated_question = [{"generated_text": "This is the pipeline output"}]
        log_file.write("Gold output: " + gold_question + "\n")
        log_file.write("rouge1: " + str(scores["rouge1"]) + "\n")
        log_file.write("rouge2: " + str(scores["rouge2"]) + "\n")
        log_file.write("rougeL: " + str(scores["rougeL"]) + "\n")
        log_file.write("rougeLsum: " + str(scores["rougeLsum"]) + "\n")
        log_file.write("~"*50)
        log_file.write("\n")
    
    log_file.write("~"*50)
    log_file.write("\n")
    log_file.write("Average rouge1: " + str(accumulated_scores["rouge1"]/n_claims) + "\n")
    log_file.write("Average rouge2: " + str(accumulated_scores["rouge2"]/n_claims) + "\n")
    log_file.write("Average rougeL: " + str(accumulated_scores["rougeL"]/n_claims) + "\n")
    log_file.write("Average rougeLsum: " + str(accumulated_scores["rougeLsum"]/n_claims) + "\n")
    log_file.write("~"*50)
    log_file.close()
generate_questions(test_data)
        





