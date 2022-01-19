
import json
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from transformers import TFAutoModelForSequenceClassification
from transformers import BertTokenizer
from transformers import InputExample, InputFeatures
from transformers import pipeline



import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

# hyperparameters sent by the client are passed as a command-line arguments to the script
parser.add_argument("--epochs",type=int,default=3)

# data, model and output directories
parser.add_argument("--output_dir",type=str,default=os.environ["SM_OUTPUT_DATA_DIR"])
parser.add_argument("--model_dir",type=str,default=os.environ["SM_MODEL_DIR"])
parser.add_argument("--training_dir",type=str,default=os.environ["SM_CHANNEL_TRAIN"])

args, _ = parser.parse_known_args()


print("GPU:", tf.config.list_physical_devices("GPU"))

label2id = {'positive': 1, 'negative': 0}
data = pd.read_csv("data/train.csv") # imdb dataset from kaggle (https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews)


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, id2label={v: k for k, v in label2id.items()}, use_cache=False)

data['sentiment'] = data['sentiment'].map(label2id)


msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data[~msk]


def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN): 
  train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)

  validation_InputExamples = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
  return train_InputExamples, validation_InputExamples
  
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, "review", "sentiment")

train_dataset = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)

validation_dataset = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_dataset = validation_dataset.batch(32)


optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
model.summary()

model.fit(
  train_dataset,
  validation_data=validation_dataset,
  epochs=1,
)



model.save_pretrained("output/model")
tokenizer.save_pretrained("output/model")

# Fix to remove the cache dependencies
tokenizer_config_file = 'output/model/tokenizer_config.json'
with open(tokenizer_config_file, 'r') as f:
  d = json.load(f)
  del d['tokenizer_file']
  f.close()

with open(tokenizer_config_file, 'w') as f:
  json.dump(d, f, indent=2)
  f.close()



sample = data.sample(100)
reviews = sample['review'].to_list()

sample

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
predictions = classifier(reviews)

y_test = sample['sentiment'].to_list()
y_pred = [label2id[pred['label']] for pred in predictions]


report = classification_report(y_test, y_pred, target_names=label2id.keys(), output_dict=True)
pd.DataFrame(report).T


cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=range(2)), range(2), range(2))
# sn.set(font_scale=1.4) # for label size
# sn.heatmap(cm, annot=True, annot_kws={"size": 16}) # font size
# plt.show()
f = open(os.path.join(args.output_dir,'report.json'),'w')
f.write(pd.DataFrame(report).T.to_json(orient='index',indent=2))
f.close()

