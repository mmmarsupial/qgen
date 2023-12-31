from transformers import AutoTokenizer
from datasets import load_metric
import numpy as np



class CalculateMetric:
    def __init__(self, model_name, metric="rouge"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.metric = load_metric(metric)

    # Computes the metric for the predicted output given the gold output
    def compute_metrics(self, pred_labs):
        
        tokenized_preds, tokenized_labs = pred_labs
        preds, labs = self._detokenize(tokenized_preds, tokenized_labs)
        scores = self.metric.compute(predictions=preds, 
                                     references=labs,
                                     use_stemmer=True)
        return {key: round(value.mid.fmeasure * 100, 2) for key, value in scores.items()}
        
    # Detokenizers using the given model tokenizer
    def _detokenize(self, tokenized_preds, tokenized_labs):

        preds = self.tokenizer.batch_decode(tokenized_preds, skip_special_tokens=True)
        tokenized_labs = np.where(tokenized_labs != -100, 
                                  tokenized_labs, 
                                  self.tokenizer.pad_token_id)
        labs = self.tokenizer.batch_decode(tokenized_labs, skip_special_tokens=True)
        return preds, labs
    
    # To compute scores for already detokenized output
    def compute_metrics_detokenized(self, pred_labs):
        preds, labs = pred_labs
        scores = self.metric.compute(predictions=preds, 
                                     references=labs,
                                     use_stemmer=True)
        return {key: round(value.mid.fmeasure * 100, 2) for key, value in scores.items()}
