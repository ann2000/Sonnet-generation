 # Reference: https://www.youtube.com/watch?v=UbCWoMf80PY&t=470s 
import numpy as np
from flask import Flask, render_template
from keras.models import load_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_sonnet',methods=['POST'])
def generate_sonnet():
    
    # Max length of each sequence. Same as in Jupyter notebook
    maxlen = 45
    # Sample new sequence every step characters
    step = 3
    data, char_indices, chars = load_data(maxlen, step)

    # Load model trained in Jupyter notebook
    model = load_model('sonnet_model.h5')
    # Choose random seed text
    start_idx = np.random.randint(0, len(data) - maxlen - 1)
    new_sonnet = data[start_idx:start_idx + maxlen]
    full_sonnet = new_sonnet
    for i in range(610):
        # Vectorize generated text
        sampled = np.zeros((1, maxlen, len(chars)))
        for j, char in enumerate(new_sonnet):
            sampled[0, j, char_indices[char]] = 1.

        # Predict next character
        preds = model.predict(sampled, verbose=0)[0]
        pred_idx = sample(preds, temperature=0.5)
        next_char = chars[pred_idx]

        # Append predicted character to seed text
        new_sonnet += next_char
        new_sonnet = new_sonnet[1:]
        full_sonnet += next_char
    
    return render_template('index.html', sonnet='{}'.format(full_sonnet))


def load_data(maxlen=45, step=3):
    
    with open('dataset.txt', 'r') as f:
        raw_data = f.read().lower()
    data = raw_data.replace("<eos>","\n")

    sentences = []
    targets = []
    # Loop through sonnets and create sequences and associated targets
    for i in range(0, len(data) - maxlen, step):
        sentences.append(data[i:i + maxlen])
        targets.append(data[maxlen + i])
    # Grab all unique characters in corpus
    chars = sorted(list(set(data)))

    # Dictionary mapping unique character to integer indices
    char_indices = dict((char, chars.index(char)) for char in chars)
    return data, char_indices, chars


def sample(preds, temperature=1.0):
   
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


if __name__ == "__main__":
    app.run(debug=True)
