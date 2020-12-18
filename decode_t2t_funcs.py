"""
Methods to encode and decode model data
"""

import tensorflow as tf
import numpy as np

'''
def encode(input_txt, encoders):
    """List of Strings to features dict, ready for inference"""
    encoded_inputs = [encoders["inputs"].encode(x) + [1] for x in input_txt]

    # pad each input so is they are the same length
    biggest_seq = len(max(encoded_inputs, key=len))
    for i, text_input in enumerate(encoded_inputs):
        encoded_inputs[i] = text_input + [0 for x in range(biggest_seq - len(text_input))]

    # Format Input Data For Model
    batched_inputs = tf.reshape(encoded_inputs, [len(encoded_inputs), -1, 1])
    return {"inputs": batched_inputs}


def decode(integers, encoders):
    """Decode list of ints to list of strings"""

    # Turn to list to remove EOF mark
    to_decode = list(np.squeeze(integers))

    if isinstance(to_decode[0], np.ndarray):
        to_decode = map(lambda x: list(np.squeeze(x)), to_decode)

    else:
        to_decode = [to_decode]

    # remove <EOF> Tag before decoding
    to_decode = map(lambda x: x[:x.index(1)], filter(lambda x: 1 in x, to_decode))

    # Decode and return Translated text
    return [encoders["inputs"].decode(np.squeeze(x)) for x in to_decode]

'''

def encode(input_str, encoders, output_str=None):
    """Input str to features dict, ready for inference"""
    inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
    batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
    return {"inputs": batch_inputs}


def decode(integers, encoders):
    """List of ints to str"""
    integers = list(np.squeeze(integers))
    if 1 in integers:
        integers = integers[:integers.index(1)]
    return encoders["inputs"].decode(np.squeeze(integers))
