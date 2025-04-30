from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(model, tokenizer, dataset):
    from transformers import Trainer
    trainer = Trainer(model=model)
    out = trainer.predict(dataset)
    preds = np.argmax(out.predictions, axis=1)
    y_true = np.array(dataset["labels"])
    print(classification_report(y_true, preds, zero_division=0))

