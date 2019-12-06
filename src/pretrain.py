from stable_baselines.common.base_class import BaseRLModel
import numpy as np
from keras import Model

class Pretrain:
    def __init__(self, model: BaseRLModel, states:np.darray, 
                 actions:np.ndarray):
        self.model = model
        self.states = states
        self.actions = actions

    #TODO: Verify that what output position is the stddev of the policy
    #TODO: make dataset actions with stddev equal to 0
    #TODO: Make clone of the model policy using keras
    #TODO: Train the cloned model
    #TODO: set the weights of the BaseRLModel
    
    def _add_stddev(self, actions: np.ndarray) -> np.ndarray:
        """The dataset action has just the actual value, the pre-trained agent
           will be trained with mean equal to those values and stddev equal to 
           0"""
        """CODE HERE"""
        pass

    def _clone_model(self, model: BaseRLModel) -> Model:
        """CODE HERE"""
        pass

    def fit(self, num_epochs: int) -> Model:
        pass

    def _set_weights_base(self) -> None:
        """'Clone the model on the way Keras -> BaseRLModel"""
        pass

