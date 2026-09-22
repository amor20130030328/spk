import time
from typing import List
import numpy as np
import mindspore_lite as mslite
import threading
class BaseLiteModel:
    def __init__(self, model_path, model_type=mslite.ModelType.MINDIR, device_type="ascend", config_path=""):
        self.model_path = model_path
        self.model_type = model_type
        self.device_type = device_type
        self.config_path = config_path
        self.context = mslite.Context()
        self.context.target = [device_type]
        self.model = mslite.Model()
        self.model.build_from_file(self.model_path, self.model_type, self.context)
        self.model_lock = threading.RLock()  # 可重入锁

    def predict(self, inputs, outputs=None) -> List[mslite.Tensor]:
        with self.model_lock:
            model_inputs:List[mslite.Tensor] = self.model.get_inputs()
            if len(model_inputs)!=len(inputs):
                raise ValueError("input data size is not equal model input size")
            shapes = [input_data_i.shape for input_data_i in inputs]
            self.model.resize(model_inputs, shapes)  # for model resize
            for i, data in enumerate(inputs):
                if isinstance(data, np.ndarray):
                    model_inputs[i].set_data_from_numpy(data)
                elif isinstance(data, mslite.Tensor):
                    model_inputs[i] = data
                else:
                    raise ValueError(f'wrong input types. type of model_input {i} is {type(model_inputs[i])}')
            if outputs:
                self.model.predict(model_inputs, outputs)  # 使用预分配的张量保存输出，device侧张量零拷贝
            else:
                outputs = self.model.predict(model_inputs)
            return outputs

    def forward(self, chunk_xs) -> List[mslite.Tensor]:
        outputs = self.model.predict([chunk_xs])
        return outputs[0].get_data_to_numpy()

if __name__ == '__main__':

    lite_base_model = BaseLiteModel(model_path="/model-data/model/w2vbert_spk_model_opt_graph.mindir")

