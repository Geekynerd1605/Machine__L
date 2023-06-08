from bert.tokenization.bert_tokenization import FullTokenizer
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import bert
import os

app = Flask(__name__)
model = tf.keras.models.load_model('my_model.h5', custom_objects={"BertModelLayer": bert.BertModelLayer})
# model = keras.models.load_model("my_model.h5")


max_seq_len = 38
classes = ['PlayMusic', 'AddToPlaylist', 'RateBook', 'SearchScreeningEvent', 'BookRestaurant', 'GetWeather',
           'SearchCreativeWork']

bert_model_name = "uncased_L-12_H-768_A-12"
bert_ckpt_dir = os.path.join("model/", bert_model_name)
tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['text']  # get the sentence input from the HTML form
    pred_tokens = tokenizer.tokenize(sentence)
    pred_tokens = ["[CLS]"] + pred_tokens + ["[SEP]"]
    pred_token_ids = tokenizer.convert_tokens_to_ids(pred_tokens)
    pred_token_ids = pred_token_ids + [0] * (max_seq_len - len(pred_token_ids))
    pred_token_ids = np.array(pred_token_ids).reshape(1, -1)
    prediction = model.predict(pred_token_ids).argmax(axis=-1)
    intent = classes[prediction[0]]
    return render_template('index.html', text="Text: {}\n".format(sentence), intent="Intent: {}".format(intent))


if __name__ == '__main__':
    app.run(debug=True)
