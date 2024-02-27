from itertools import chain
from functools import partial

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from data import tokenize
from utils import compute_metrics

import json
import numpy as np


TRAINING_MODEL_PATH = "albert/albert-base-v2"
TRAINING_MAX_LENGTH = 128
OUTPUT_DIR = "output"


data = json.load(open("data/train.json"))

# downsampling of negative examples
p = []  # positive samples (contain relevant labels)
n = (
    []
)  # negative samples (presumably contain entities that are possibly wrongly classified as entity)
for d in data:
    if any(np.array(d["labels"]) != "O"):
        p.append(d)
    else:
        n.append(d)
print("original datapoints: ", len(data))

external = json.load(open("data/pii_dataset_fixed.json"))
print("external datapoints: ", len(external))

moredata = json.load(open("data/moredata_dataset_fixed.json"))
print("moredata datapoints: ", len(moredata))

data = moredata + external + p + n[: len(n) // 3]
print("combined: ", len(data))


all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i, l in enumerate(all_labels)}
id2label = {v: k for k, v in label2id.items()}

target = [
    "B-EMAIL",
    "B-ID_NUM",
    "B-NAME_STUDENT",
    "B-PHONE_NUM",
    "B-STREET_ADDRESS",
    "B-URL_PERSONAL",
    "B-USERNAME",
    "I-ID_NUM",
    "I-NAME_STUDENT",
    "I-PHONE_NUM",
    "I-STREET_ADDRESS",
    "I-URL_PERSONAL",
]
print(id2label)


tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)

ds = Dataset.from_dict(
    {
        "full_text": [x["full_text"] for x in data],
        "document": [str(x["document"]) for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        "provided_labels": [x["labels"] for x in data],
    }
)
ds = ds.map(
    tokenize,
    fn_kwargs={
        "tokenizer": tokenizer,
        "label2id": label2id,
        "max_length": TRAINING_MAX_LENGTH,
    },
    num_proc=3,
)
# ds = ds.class_encode_column("group")


model = AutoModelForTokenClassification.from_pretrained(
    TRAINING_MODEL_PATH,
    num_labels=len(all_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)


final_ds = ds.train_test_split(
    test_size=0.2, seed=42
)  # cannot use stratify_by_column='group'
final_ds


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    fp16=True,
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    report_to="none",
    evaluation_strategy="no",
    do_eval=False,
    save_total_limit=1,
    logging_steps=20,
    lr_scheduler_type="cosine",
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_ratio=0.1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=partial(compute_metrics, all_labels=all_labels),
)


trainer.train()


trainer.save_model("output/albert-base-v2_128")
tokenizer.save_pretrained("output/albert-base-v2_128")
