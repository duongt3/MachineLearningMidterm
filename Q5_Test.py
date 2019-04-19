"""
PS#2
Q5 (Testing) - A Small Character Level LSTM
Loads a trained LSTM and mapping and generates sentences

"""

from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

seed_text = "Some of his King's Bench familiars, who were occasionally parties to the full-bodied wine and the lie, excused him"
n_chars_to_predict = 500
seq_length = 100

# load the model and mapping
model = load_model('LargeLSTM_model_256_4096_100.h5')
mapping = load(open('LargeLSTM_mapping.pkl', 'wb'))


# Make predictions
for k in range(n_chars_to_predict):
    # encode the characters as integers
    encoded = [mapping[char] for char in seed_text]
    # truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    # one hot encode
    encoded = to_categorical(encoded, num_classes=len(mapping))
    # predict character
    yhat = model.predict_classes(encoded, verbose=0)
    
    # reverse map integer to character
    for char, index in mapping.items():
        if index == yhat:
            break
    seed_text += char

print(seed_text)
