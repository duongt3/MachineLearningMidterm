"""
PS#2
Q5 (Testing) - A Small Character Level LSTM
Loads a trained LSTM and mapping and generates sentences

"""

from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

seed_text = "the passenger booked by this history was on the coachstep getting in the two other passengers were close behind him and about to follow. he remained on the step half in the coach and half out of they remained"
n_chars_to_predict = 500
seq_length = 100

# load the model and mapping
model = load_model('LargeLSTM_model_512_4096_50.h5')
mapping = load(open('LargeLSTM_mapping.pkl', 'r'))

# Make predictions
for k in range(n_chars_to_predict):
    # encode the characters as integers
    encoded = [mapping[char] for char in seed_text]
    # truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    # one hot encode
    encoded = encoded / float(len(mapping))
    encoded = np.reshape(encoded, (encoded.shape[0], length, 1))
    # predict character
    yhat = model.predict_classes(encoded, verbose=0)
    
    # reverse map integer to character
    for char, index in mapping.items():
        if index == yhat:
            break
    seed_text += char

print(seed_text)
