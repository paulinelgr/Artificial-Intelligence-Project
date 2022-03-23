import numpy as np
import tensorflow as tf
from tensorflow import keras

# Init your variables here 

# Put your name Here 
name = "Pauline"

BRAKE = 0
ACCELERATE = 1
LEFT5 = 2
RIGHT5 = 3

model = keras.models.load_model('drivers/entrainement_1')

def drive(d1, d2, d3, d4, d5):
    # d1  front
    # d2  mid right
    # d3  mid left
    # d4  right
    # d5  left

    # List of possible actions to return
    # BRAKE
    # ACCELERATE
    # LEFT5
    # RIGHT5

    state = [d1, d2, d3, d4, d5]
    state = np.reshape(state, [1, 5])
    act_values = model.predict(state)
    return np.argmax(act_values[0])

