 # Reference: https://www.youtube.com/watch?v=UbCWoMf80PY&t=470s 
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from pickle import load
import fnmatch, os, shutil
from werkzeug.utils import secure_filename



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_sonnet',methods=['POST']) 
# extract features from each photo in the directory

def generate_sonnet():

    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
    
    # load the tokenizer
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    # pre-define the max sequence length (from training)
    max_length = 34
    # load the model
    model = load_model('image_context.h5')
    # load and prepare the photograph
    photo = extract_features(f.filename)
    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)
    description = list(description.split(" "))
    description = [i for i in description if i != "the"]
    description = " ".join(description)
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
    for i in range(580):
        if (i==500):
            full_sonnet = full_sonnet+description
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

def extract_features(filename):

  
	# load the model
	model = VGG16()
	# re-structure the model
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature
 
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = ''
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
	return in_text

def load_data(maxlen=30, step=3):
    
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
