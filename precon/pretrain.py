from stable_baselines.common.base_class import BaseRLModel
import numpy as np
from keras import Model
from precon.keras_from_rl import keras_from_rl


class Pretrain:
    def __init__(self, model: BaseRLModel, states: np.ndarray,
                 actions: np.ndarray):
        self.model = model
        self.states = states
        self.actions = actions
        self.keras_model = keras_from_rl(model)

    def fit(self, num_epochs: int, batch_size=64, validation_split=0.75,
            verbose=0) -> Model:

        self.keras_model.compile(optimizer="adam", loss="mse",
                                 metrics=["mse"])
        self.keras_model.fit(self.states, self.actions,
                             batch_size=batch_size,
                             epochs=num_epochs,
                             validation_split=validation_split,
                             verbose=verbose)

    def get_pretrained_model(self) -> BaseRLModel:
        """'Clone the model on the way Keras -> BaseRLModel"""
        keys = {}
        keras_parameters = self.keras_model.get_weights()
        rl_parameters = self.model.get_parameters()
        index = 0
        for parameter in rl_parameters:
            cond1 = parameter.startswith("model/pi")
            cond2 = not parameter.startswith('model/pi/logstd')
            if cond1 and cond2:
                keys[parameter] = keras_parameters[index]
                index += 1
        self.model.load_parameters(keys, exact_match=False)
        return self.model
        

