import pandas as pd
from transformers import AutoTokenizer
from secondary.prepare_data import DataPrep
from secondary.metric import CalculateMetric
from sys import argv, exit
from getopt import getopt

def main(argv):
    print()
    print("Fine-tune a model for question generation given a claim")
    print()

    model_name = "t5-small"                                     # default
    output_dir = "./checkpoints/" + model_name + "-fine-tuned"  # default
    use_full_data = True                                        # default


    opts, _ = getopt(argv, "hi:o:f")
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
    
    print("Pre-trained model: ", model_name)
    print("Output path is: ", output_dir)
    print("Uses full data: ", use_full_data)

if __name__ == "__main__":
    main(argv[1:])