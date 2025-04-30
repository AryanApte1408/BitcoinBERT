from transformers import BertForSequenceClassification, TrainingArguments, Trainer
import config

def train_model(train_ds, val_ds):
    model = BertForSequenceClassification.from_pretrained(config.BERT_MODEL, num_labels=len(config.LABELS))

    args = TrainingArguments(
        output_dir="bitcoin-bert",
        overwrite_output_dir=True,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        do_train=True,
        do_eval=True,
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=metrics
    )
    trainer.train()
    trainer.save_model("bitcoin-bert-final")
    return model

def metrics(p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    preds = p.predictions.argmax(axis=1)
    acc = accuracy_score(p.label_ids, preds)
    p_, r, f, _ = precision_recall_fscore_support(p.label_ids, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": p_, "recall": r, "f1": f}
