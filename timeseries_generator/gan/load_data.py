import os
import numpy as np
import pickle

from .util import add_gen_flag
from .util import normalize_feature_per_sample
from .util import normalize_attribute
from .util import normalize_auxl_signal

class DataLoader():
    def __init__(self, args, data_path):
        self.data_path = data_path
        self.addi_attr_power_transform = args.addi_attr_power_transform
        self.dataset_name = args.dataset_name
        self.sample_len = args.sample_len
        self.verbose_summary = args.verbose_summary
        self.train_with_auxl_signal = args.train_with_auxl_signal

    def load_data(self):
        self.read_dataset()
        self.preprocess_dataset()
        self.check_data()
        if self.verbose_summary:
            self.dataset_summary()
        return (self.data_feature, self.data_attribute, self.data_gen_flag,
                self.data_feature_outputs, self.data_attribute_outputs,
                self.real_attribute_mask, self.data_attribute_real,
                self.data_attribute_scalers,
                self.data_auxl_signal, self.data_feature_ground_truth
                ) 

    # region Load original dataset
    def read_dataset(self):
        data_npz = np.load(
            os.path.join(self.data_path, "data_{}.npz".format(self.dataset_name)))
        with open(os.path.join(self.data_path, "data_feature_output.pkl"), "rb") as f:
            self.data_feature_outputs = pickle.load(f)
        with open(os.path.join(self.data_path,
                            "data_attribute_output.pkl"), "rb") as f:
            self.data_attribute_outputs = pickle.load(f)

        self.data_feature = data_npz["data_feature"]
        self.data_feature_ground_truth = data_npz["data_feature"]
        self.data_attribute = data_npz["data_attribute"]
        self.data_gen_flag = data_npz["data_gen_flag"]

        if "data_auxl_signal" in data_npz:
            with open(os.path.join(self.data_path,
                            "data_auxl_signal_output.pkl"), "rb") as f:
                self.data_auxl_signal_outputs = pickle.load(f)
            self.data_auxl_signal = data_npz["data_auxl_signal"]
        else:
            self.data_auxl_signal = None
            self.data_auxl_signal_outputs = None
    # endregion

    def preprocess_dataset(self):
        # region Normalise dataset
        (self.data_feature, self.data_attribute,
         self.data_attribute_outputs, self.real_attribute_mask) = \
            normalize_feature_per_sample(
            self.data_feature, self.data_attribute,
            self.data_feature_outputs, self.data_attribute_outputs
            )
        # endregion

        # region Add gen_flag
        (self.data_feature, self.data_feature_outputs) = \
            add_gen_flag(self.data_feature, self.data_gen_flag,
                         self.data_feature_outputs, self.sample_len
                         )
        # endregion

        # region Normalise attribute
        (self.data_attribute,
         self.data_attribute_scalers) = \
            normalize_attribute(self.data_attribute,
                                self.data_attribute_outputs,
                                self.real_attribute_mask,
                                self.addi_attr_power_transform)
        
        self.data_attribute_real = self.data_attribute[:, :-2]
        # endregion

        # region Normalise auxl_signal
        if self.data_auxl_signal is not None:
            (self.data_auxl_signal) = \
                normalize_auxl_signal(self.data_auxl_signal,
                                      self.data_auxl_signal_outputs)
        # endregion
    
    # region Check data
    def check_data(self):
        self.gen_flag_dims = []
        self.gen_flag_dims = next(
            ([dim, dim + 1] for dim, output 
             in enumerate(self.data_feature_outputs) 
             if output.is_gen_flag and output.dim == 2),
            None
        )
        if self.gen_flag_dims is None:
            raise Exception("gen flag not found or its dim is not 2")
        if len(self.gen_flag_dims) == 0:
            raise Exception("gen flag not found")
        if (self.data_feature.shape[2] !=
                np.sum([t.dim for t in self.data_feature_outputs])):
            raise Exception(
                "feature dimension does not match data_feature_outputs")
        if len(self.data_gen_flag.shape) != 2:
            raise Exception("data_gen_flag should be 2 dimension")
        self.data_gen_flag = np.expand_dims(self.data_gen_flag, 2)

        if self.data_feature.shape[1] % self.sample_len != 0:
            raise Exception("length must be a multiple of sample_len")
        
        tolerance = 1e-3
        if self.data_feature.min() < -1-tolerance or self.data_feature.max() > 1+tolerance:
            print("Warning: data_feature values are not normalised")
            
        if self.data_attribute.min() < -1-tolerance or self.data_attribute.max() > 1+tolerance:
            print("Warning: data_attribute values are not normalised")

        if self.train_with_auxl_signal:
            assert self.data_auxl_signal is not None, "data_auxl_signal must be provided to train with auxiliary signal"
            if (self.data_auxl_signal.shape[0] !=
                    self.data_feature.shape[0]):
                raise Exception(
                    "data_auxl_signal and data_feature should have same length")
    # endregion
    
    # region Dataset summary
    def dataset_summary(self):
        print("============Dataset Summary==============")
        print("data feature shape: {}".format(self.data_feature.shape))
        print("data feature outputs length: {}".format(len(self.data_feature_outputs)))
        print("=========================================")
        print("data attribute shape: {}".format(self.data_attribute.shape))
        print("data attribute outputs length: {}".format(len(self.data_attribute_outputs)))
        print("real_attribute_mask:{}".format(self.real_attribute_mask))
        print("data attribute real shape:{}".format(self.data_attribute_real.shape))
        print("=========================================")
        print("data gen flag shape: {}".format(self.data_gen_flag.shape))
        print("=========================================")
        if self.data_auxl_signal is not None:
            print("data auxl signal shape: {}".format(self.data_auxl_signal.shape))
            print("data auxl signal outputs length: {}".format(len(self.data_auxl_signal_outputs)))
            print("=========================================")
    # endregion
    
