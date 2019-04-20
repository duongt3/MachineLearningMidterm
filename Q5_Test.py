"""
PS#2
Q5 (Testing) - A Small Character Level LSTM
Loads a trained LSTM and mapping and generates sentences

"""

from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

seed_text = "The passenger booked by this history was on the coachstep, getting in the two other passengers were close behind him and about to follow. He remained on the step half in the coach and half out of they remained"
seed_text = seed_text + "in the road below him. They all looked from the coachman to the guard and from the guard to the coachman and listened. The coachman looked back and the guard looked back and even the emphatic leader pricked up his ears and looked back without contradicting."
n_chars_to_predict = 500
seq_length = 100

# load the model and mapping
model = load_model('LargeLSTM_model_256_4096_100.h5')
mapping = load(open('LargeLSTM_mapping.pkl', 'r'))

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
