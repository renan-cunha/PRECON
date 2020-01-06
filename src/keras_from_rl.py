import keras
from stable_baselines.common.base_class import BaseRLModel
from typing import Dict, List
import numpy as np


def keras_from_rl(model: BaseRLModel) -> keras.Model:

    layers = get_policy_shape(model)
    model = create_model(layers)

    return model


def get_policy_shape(model: BaseRLModel) -> List[int]:
    """Returns a list with the number of neurons in each layer of the
    BaseRLModel"""
    output = []
    output.append(model.observation_space.shape[0])
    parameters_dict = model.get_parameters()
    for key in parameters_dict:
        cond1 = key.startswith('model/pi')
        cond2 = key.startswith('model/pi/logstd')
        cond3 = key.endswith("b:0")
        if cond1 and not cond2 and cond3:
            dim = parameters_dict[key].shape[0]
            output.append(dim)
    return output


def create_model(layers: List[int]) -> keras.Model:
    model = keras.Sequential()
    input_dim = layers[0]
    first_layer = layers[1]
    model.add(keras.layers.Dense(first_layer, input_dim=input_dim,
                                 activation="tanh"))
    for num_neurons in layers[2:]:
        model.add(keras.layers.Dense(num_neurons, activation="tanh"))
    return model
