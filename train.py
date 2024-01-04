# External imports
import torch
import nltk
from sys import (argv,
                  exit)
from getopt import getopt
from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM, 
                          DataCollatorForSeq2Seq, 
                          Seq2SeqTrainer, 
                          Seq2SeqTrainingArguments)

# Internal imports
from secondary.prepare_data import DataPrep
from secondary.metric import CalculateMetric

def main(argv):
    print()
    print("Fine-tune a model for question generation given a claim")
    print()

    # Variables with default values: Possible to change using flags
    model_name = "t5-small"                                     # default. flag: -i
    output_dir = "./checkpoints/" + model_name + "-fine-tuned"  # default. flag: -o
    use_full_data = True                                        # default. flag: -f
    learning_rate = 0.00005                                     # default. flag: -l
    batch_size = 16                                             # default. flag: -b


    opts, _ = getopt(argv, "hi:o:fl:b:")
    for opt, arg in opts:
        if opt == "-h":
            print("Usage: train.py [-h] [-i <base pre-trained model>] [-o <output path>]")
            print ("-h\t\tPrints usage information.")
            print ("-i <model>\tSets pre-trained model. Default is t5-small.")
            print ("-o <model>\tSets the output path for the fine-tuned model")
            print ()
            exit()
        elif opt == "-i":
            model_name = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-f":
            use_full_data = False
        elif opt == "-l":
            learning_rate = arg
        elif opt == "-b":
            batch_size = arg
    
    print("Pre-trained model: ", model_name)
    print("Output path is: ", output_dir)
    print("Uses full data: ", use_full_data)
    print("Learning rate: ", learning_rate)
    print("Batch size: ", batch_size)

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


    # Load and prepare train and validation data
    data_prep = DataPrep(model_name)
    train_data = data_prep.prepare_data(paths["train"])
    validation_data = data_prep.prepare_data(paths["validation"])

    # Model, tokenizer, metric and collator initiation
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    metric = CalculateMetric(tokenizer=tokenizer, metric="rouge")
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    # Training arguments
    args = Seq2SeqTrainingArguments(
        learning_rate=learning_rate,                                                # Default 1e-5 for BART. Add warmup???
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=200,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=5,
        predict_with_generate=True,
        #fp16=True,
        load_best_model_at_end=True,
        output_dir=output_dir
        )

    # Trainer 
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metric.compute_metrics
        )

    # Train and save model
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main(argv[1:])