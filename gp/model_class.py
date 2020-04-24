import numpy as np
import run_params_ret as run_params
from celerite.modeling import Model


class CustomModel(Model):


    parameter_names=tuple(run_params.ret_params)


    def get_value(self, y=None,  retrieval=None):
        # print(ret_params)
        ret_params=self.get_parameter_dict()

        result=retrieval.get_model(ret_params=ret_params)

        return result
