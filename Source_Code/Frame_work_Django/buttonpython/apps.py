from django.apps import AppConfig
import html
import pathlib
import os

from fast_bert.prediction import BertClassificationPredictor

class WebappConfig(AppConfig):
    name = 'fastbert'
    MODEL_PATH = Path("model")
    BERT_PRETRAINED_PATH = Path("model/")
    LABEL_PATH = Path("label/")
    predictor = BertClassificationPredictor(model_path = MODEL_PATH/"multilabel-emotion-fastbert-basic.bin", 
                                            pretrained_path = BERT_PRETRAINED_PATH, 
                                            label_path = LABEL_PATH, multi_label=True) 
