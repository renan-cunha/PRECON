from stable_baselines.common.base_class import BaseRLModel
import numpy as np
from keras import Model
from src.keras_from_rl import keras_from_rl


class Pretrain:
    def __init__(self, model: BaseRLModel, states: np.ndarray,
                 actions: np.ndarray):
        self.model = model
        self.states = states
        self.actions = actions
        self.keras_model = keras_from_rl(model)

    #TODO: Train the cloned model
    #TODO: set the weights of the BaseRLModel

    def fit(self, num_epochs: int, batch_size=64, validation_split=0.75) -> Model:

        self.keras_model.compile(optimizer="adam", loss="mse",
                                 metrics=["mse"])
        self.keras_model.fit(self.states, self.actions,
                             batch_size=batch_size,
                             epochs=num_epochs,
                             validation_split=validation_split)

    def get_pretrained_model(self) -> BaseRLModel:
        """'Clone the model on the way Keras -> BaseRLModel"""
        pass

