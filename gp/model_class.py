import numpy as np
import run_params_ret as run_params
from celerite.modeling import Model
from retrieval import Retrieval


class CustomModel(Model):
    parameter_names = tuple(run_params.ret_params)

    def get_value(self, wavelength, retrieval=Retrieval()):
        ret_params = self.get_parameter_dict()
        result = retrieval.get_model(wavelength, ret_params=ret_params)
        return result
