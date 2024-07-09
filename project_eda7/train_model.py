import re

import evaluate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict, Translation
from transformers.keras_callbacks import PushToHubCallback, KerasMetricCallback
from transformers import pipeline, AutoTokenizer, AdamWeightDecay, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq


HF_DATASET_NAME = "uvci/Koumankan_mt_dyu_fr"
SRC_LANG = "dyu"
TRG_LANG = "fr"
CHARS_TO_REMOVE_REGEX = '[!"&\(\),-./:;=?+.\n\[\]]'
PREFIX = "translate Dyula to French: " # This command will have to be passed to the model during inference so it knows what to do
MODEL_CHECKPOINT = "t5-small"
HF_USERNAME = "Mluleki"
HF_REPO_NAME = "dyu-fr-translation"
LOCAL_SAVE_DIR = "saved_model"
TRAIN_EPOCHS = 10

dataset = load_dataset(HF_DATASET_NAME)


def remove_special_characters(text):
    text = re.sub(CHARS_TO_REMOVE_REGEX, " ", text.lower())
    return text.strip()

def clean_text(batch):
    # process source text
    batch["translation"][SRC_LANG] = remove_special_characters(batch["translation"][SRC_LANG])
    # process target text
    batch["translation"][TRG_LANG] = remove_special_characters(batch["translation"][TRG_LANG])
    return batch

dataset = dataset.map(clean_text)

def preprocess_function(examples):
    inputs = [PREFIX + example[SRC_LANG] for example in examples["translation"]]
    targets = [example[TRG_LANG] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs


tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
tokenized_books = dataset.map(preprocess_function, batched=True)


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL_CHECKPOINT, return_tensors="tf")

optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)


tf_train_set = model.prepare_tf_dataset(
    tokenized_books["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    tokenized_books["validation"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)


model.compile(optimizer=optimizer)  # No loss argument!

push_to_hub_callback = PushToHubCallback(
    output_dir=LOCAL_SAVE_DIR,
    hub_model_id=f"{HF_USERNAME}/{HF_REPO_NAME}",
    tokenizer=tokenizer,
)
callbacks = [push_to_hub_callback] # metric_callback

train_hist = model.fit(
  x=tf_train_set,
  # validation_split = 0.2,
  validation_data=tf_test_set,
  epochs=TRAIN_EPOCHS,
  callbacks=callbacks
)

plt.figure(figsize=(6,3))
plt.plot(train_hist.history["loss"], label="Training loss")
plt.plot(train_hist.history["val_loss"], label="Validation loss")
plt.title("Train and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig('train.jpg')

text = f"{PREFIX}i tɔgɔ bi cogodɔ"

translator = pipeline("translation", model=f"{HF_USERNAME}/{HF_REPO_NAME}")
print(translator(text))


tokenizer.save_pretrained(LOCAL_SAVE_DIR)
model.save_pretrained(LOCAL_SAVE_DIR)

## Do everything in one step using pipeline
# from transformers import pipeline

translator = pipeline("translation", model=LOCAL_SAVE_DIR)

print(translator(text))


tokenizer = AutoTokenizer.from_pretrained(LOCAL_SAVE_DIR)
loaded_model = TFAutoModelForSeq2SeqLM.from_pretrained(LOCAL_SAVE_DIR)

inputs = tokenizer(text, return_tensors="tf").input_ids
outputs = loaded_model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95, temperature=1.0)
tokenizer.decode(outputs[0], skip_special_tokens=True)

inputs = tokenizer(f"{PREFIX}puɛn saba fɔlɔ", return_tensors="tf").input_ids
outputs = loaded_model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=20, top_p=0.7, temperature=0.2)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

