from .output import OutputType, Normalization
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum

# region Primary Discriminator
class PrimDiscriminator(nn.Module):
    def __init__(self, input_dim, num_layers, num_units):
        super(PrimDiscriminator, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.layers = nn.ModuleList()

        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(input_dim if i == 0
                                         else num_units, num_units))
            self.layers.append(nn.ReLU())
            # Uncomment if batch normalization is needed
            # if i > 0:
            #     self.layers.append(nn.BatchNorm1d(num_units))
        self.layers.append(nn.Linear(num_units, 1))

    def forward(self, feature, attribute):
        x1 = feature.view(feature.size(0), -1)
        x2 = attribute.view(attribute.size(0), -1)
        x = torch.cat([x1, x2], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(1)
# endregion


# region Auxiliary Discriminator
class AuxlDiscriminator(nn.Module):
    def __init__(self, input_dim, num_layers, num_units):
        super(AuxlDiscriminator, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.layers = nn.ModuleList()

        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(input_dim if i == 0
                                         else num_units, num_units))
            self.layers.append(nn.ReLU())
            # Uncomment if batch normalization is needed
            # if i > 0:
            #     self.layers.append(nn.BatchNorm1d(num_units))
        self.layers.append(nn.Linear(num_units, 1))

    def forward(self, attribute):
        x = attribute.view(attribute.size(0), -1)
        for layer in self.layers:
            x = layer(x)   
        return x.squeeze(1)
# endregion


class RNNInitialStateType(Enum):
    ZERO = "ZERO"
    RANDOM = "RANDOM"
    VARIABLE = "VARIABLE"


# region demAEndGANTimeGenerator
class demAEndGANTimeGenerator(nn.Module):
    def __init__(self, device,
                 attribute_outputs, feature_outputs, real_attribute_mask,
                 sample_len, feature_dim2,
                 feedback, noise,
                 attribute_latent_dim, attribute_num_layers, attribute_num_units,
                 feature_latent_dim, feature_num_layers, feature_num_units,
                 auxl_signal_dim2=None,
                 initial_state_type=RNNInitialStateType.RANDOM,
                 initial_stddev=0.02):
        super(demAEndGANTimeGenerator, self).__init__()

        self.device = device
        self.attribute_outputs = attribute_outputs
        self.feature_outputs = feature_outputs
        self.real_attribute_mask = real_attribute_mask
        self.sample_len = sample_len
        self.feature_dim2 = feature_dim2
        self.feedback = feedback
        self.noise = noise
        self.attribute_latent_dim = attribute_latent_dim
        self.attribute_num_layers = attribute_num_layers
        self.attribute_num_units = attribute_num_units
        self.feature_latent_dim = feature_latent_dim
        self.feature_num_layers = feature_num_layers
        self.feature_num_units = feature_num_units
        self.auxl_signal_dim2 = auxl_signal_dim2
        self.initial_state_type = initial_state_type
        self.initial_stddev = initial_stddev

        self.epsilon = 1e-5
        self.momentum = 0.9
        self.STR_REAL = "real"
        self.STR_ADDI = "addi"

        if not self.noise and not self.feedback:
            raise Exception("noise and feedback should have at least one True")

        self.feature_out_dim = (np.sum([t.dim for t in feature_outputs]) *
                                self.sample_len)
        self.attribute_out_dim = np.sum([t.dim for t in attribute_outputs])   
                
        self.real_attribute_outputs = []
        self.addi_attribute_outputs = []
        self.real_attribute_out_dim = 0
        self.addi_attribute_out_dim = 0
        for i in range(len(self.real_attribute_mask)):
            if self.real_attribute_mask[i]:
                self.real_attribute_outputs.append(self.attribute_outputs[i])
                self.real_attribute_out_dim += self.attribute_outputs[i].dim
            else:
                self.addi_attribute_outputs.append(self.attribute_outputs[i])
                self.addi_attribute_out_dim += self.attribute_outputs[i].dim

        for i in range(len(self.real_attribute_mask) - 1):
            if not self.real_attribute_mask[i] and \
                    self.real_attribute_mask[i + 1]:
                raise Exception("Real attribute should come first")

        self.gen_flag_id = None
        for i in range(len(self.feature_outputs)):
            if self.feature_outputs[i].is_gen_flag:
                self.gen_flag_id = i
                break
        if self.gen_flag_id is None:
            raise Exception("cannot find gen_flag_id")
        if self.feature_outputs[self.gen_flag_id].dim != 2:
            raise Exception("gen flag output's dim should be 2")

        # region Real attribute generator MLP and Fake min-max generator MLP
        if len(self.addi_attribute_outputs) > 0:
            all_attribute_outputs = [self.real_attribute_outputs,
                                     self.addi_attribute_outputs]
        else:
            all_attribute_outputs = [self.real_attribute_outputs]

        self.mlp_attr_input = nn.ModuleDict({
            self.STR_REAL: nn.ModuleList(),
            self.STR_ADDI: nn.ModuleList()
        })
        self.mlp_attr_output = nn.ModuleDict({
            self.STR_REAL: nn.ModuleDict({
                f"output_{a}": nn.ModuleList()
                for a in range(len(self.real_attribute_outputs))
            }),
            self.STR_ADDI: nn.ModuleDict({
                f"output_{b}": nn.ModuleList()
                for b in range(len(self.addi_attribute_outputs))
            })
        })
        
        for i, (name, layer) in enumerate(self.mlp_attr_input.items()):
            layers_in = nn.ModuleList()
            if name == self.STR_REAL:
                mlp_input_dim = self.attribute_latent_dim
            else:
                mlp_input_dim = (self.attribute_latent_dim +
                                 self.real_attribute_out_dim)

            for j in range(self.attribute_num_layers - 1):
                layers_in.append(nn.Linear(mlp_input_dim if j == 0
                                           else self.attribute_num_units,
                                           self.attribute_num_units))
                layers_in.append(nn.ReLU())
                layers_in.append(nn.BatchNorm1d(self.attribute_num_units,
                                                momentum=self.momentum,
                                                eps=self.epsilon))
            self.mlp_attr_input[name] = layers_in

            for k in range(len(all_attribute_outputs[i])):
                layers_out = nn.ModuleList()
                output = all_attribute_outputs[i][k]
                att_output_dim = output.dim
                layers_out.append(nn.Linear(self.attribute_num_units,
                                            att_output_dim))
                self.mlp_attr_output[name][f"output_{k}"] = layers_out
        # endregion

        # region Feature generator RNN
        rnn_input_dim = self.attribute_out_dim
        if self.auxl_signal_dim2 is not None:
            rnn_input_dim += self.auxl_signal_dim2
        if self.noise:
            rnn_input_dim += self.feature_latent_dim
        if self.feedback:
            rnn_input_dim += (self.sample_len * self.feature_dim2)
        self.rnn_network = nn.ModuleList()
        for i in range(self.feature_num_layers):
            self.rnn_network.append(nn.LSTMCell(input_size=rnn_input_dim
                                                if i == 0
                                                else self.feature_num_units,
                                                hidden_size=self.feature_num_units))
        
        num_rnn_output = self.sample_len * len(self.feature_outputs)
        self.rnn_output = nn.ModuleDict({
            f"output_{c}": nn.ModuleList()
            for c in range(num_rnn_output)
        })
        for m in range(num_rnn_output):
            layers_rnn_out = nn.ModuleList()
            feature_output = self.feature_outputs[m % len(
                self.feature_outputs)]
            feature_output_dim = feature_output.dim
            layers_rnn_out.append(nn.Linear(self.feature_num_units,
                                            feature_output_dim))
            self.rnn_output[f"output_{m}"] = layers_rnn_out
        # endregion

    # region Initialise RNN hidden states
    def init_rnn_state(self):
        initial_states = []
        for _ in range(self.feature_num_layers):
            if self.initial_state_type == RNNInitialStateType.ZERO:
                initial_states.append((
                    torch.zeros(self.batch_size,
                                self.feature_num_units).to(self.device),
                    torch.zeros(self.batch_size,
                                self.feature_num_units).to(self.device),
                ))
            elif self.initial_state_type == RNNInitialStateType.RANDOM:
                initial_states.append((
                    torch.randn(self.batch_size,
                                self.feature_num_units).to(self.device),
                    torch.randn(self.batch_size,
                                self.feature_num_units).to(self.device),
                ))
            elif self.initial_state_type == RNNInitialStateType.VARIABLE:
                h = nn.Parameter(torch.randn(1, self.feature_num_units) *
                                self.initial_stddev).to(self.device)
                h = h.repeat(self.batch_size, 1)
                c = nn.Parameter(torch.randn(1, self.feature_num_units) *
                                self.initial_stddev).to(self.device)
                c = c.repeat(self.batch_size, 1)
                initial_states.append((h, c))
            else:
                raise NotImplementedError
        return initial_states
    # endregion

    # region Forward for demAEndGANTimeGenerator
    def forward(self, attribute_input_noise,
                addi_attribute_input_noise,
                feature_input_noise,
                feature_input_data,
                auxl_signal=None,
                train=True,
                attribute=None,
                ):
        self.batch_size = feature_input_noise.size(0)
        # region Attribute generator and min-max generator
        if attribute is None:
            all_attribute = []
            all_discrete_attribute = []
            if len(self.addi_attribute_outputs) > 0:
                all_attribute_input_noise = [attribute_input_noise,
                                             addi_attribute_input_noise]
                all_attribute_outputs = [self.real_attribute_outputs,
                                         self.addi_attribute_outputs]
                all_attribute_part_name = [self.STR_REAL, self.STR_ADDI]
                all_attribute_out_dim = [self.real_attribute_out_dim, 
                                         self.addi_attribute_out_dim]
            else:
                all_attribute_input_noise = [attribute_input_noise]
                all_attribute_outputs = [self.real_attribute_outputs]
                all_attribute_part_name = [self.STR_REAL]
                all_attribute_out_dim = [self.real_attribute_out_dim]
        else:
            all_attribute = [attribute]
            all_discrete_attribute = [attribute]
            if len(self.addi_attribute_outputs) > 0:
                all_attribute_input_noise = [addi_attribute_input_noise]
                all_attribute_outputs = [self.addi_attribute_outputs]
                all_attribute_part_name = [self.STR_ADDI]
                all_attribute_out_dim = [self.addi_attribute_out_dim]
            else:
                all_attribute_input_noise = []
                all_attribute_outputs = []
                all_attribute_part_name = []
                all_attribute_out_dim = []

        for part_i in range(len(all_attribute_input_noise)):
            if all_attribute_part_name[part_i] == self.STR_REAL:
                attributes = all_attribute_input_noise[part_i]
            elif all_attribute_part_name[part_i] == self.STR_ADDI:
                attributes = torch.cat([all_attribute_input_noise[part_i]] +
                                       all_discrete_attribute, dim=1)

            for _, layer in enumerate(self.mlp_attr_input
                                      [all_attribute_part_name[part_i]]):
                attributes = layer(attributes)
                
            part_attribute = []
            part_discrete_attribute = []
            for i in range(len(all_attribute_outputs[part_i])):
                output = all_attribute_outputs[part_i][i]
                for _, layer in enumerate(self.mlp_attr_output
                                          [all_attribute_part_name[part_i]]
                                          [f"output_{i}"]):
                    sub_output_ori = layer(attributes)
                if output.type_.value == OutputType.DISCRETE.value:
                    sub_output = F.softmax(sub_output_ori, dim=1)
                    sub_output_discrete = F.one_hot(torch.argmax(sub_output,
                                                                 dim=1),
                                                    output.dim)
                elif output.type_.value == OutputType.CONTINUOUS.value:
                    if (output.normalization.value ==
                        Normalization.ZERO_ONE.value):
                        sub_output = torch.sigmoid(sub_output_ori)
                    elif (output.normalization.value ==
                          Normalization.MINUSONE_ONE.value):
                        sub_output = torch.tanh(sub_output_ori)
                    else:
                        raise Exception("unknown normalization type")
                    sub_output_discrete = sub_output
                else:
                    raise ValueError(f"unknown output type: {output.type_}")
                part_attribute.append(sub_output)
                part_discrete_attribute.append(sub_output_discrete)

            part_attribute = torch.cat(part_attribute, dim=1)
            part_discrete_attribute = torch.cat(part_discrete_attribute, dim=1)
            part_attribute = part_attribute.view(self.batch_size,
                                                 all_attribute_out_dim[part_i])
            part_discrete_attribute = \
                part_discrete_attribute.view(self.batch_size,
                                             all_attribute_out_dim[part_i])

            part_discrete_attribute = part_discrete_attribute.detach()

            all_attribute.append(part_attribute)
            all_discrete_attribute.append(part_discrete_attribute)

        all_attribute = torch.cat(all_attribute, dim=1)
        all_discrete_attribute = torch.cat(all_discrete_attribute, dim=1)
        all_attribute = all_attribute.view(self.batch_size,
                                           self.attribute_out_dim)
        all_discrete_attribute = all_discrete_attribute.view(
            self.batch_size,
            self.attribute_out_dim)
        # endregion

        # region Feature generator
        feature_input_data_dim = len(feature_input_data.size())
        if feature_input_data_dim == 3:
            feature_input_data_reshape = feature_input_data.permute(1, 0, 2)
        feature_input_noise_reshape = feature_input_noise.permute(1, 0, 2)
        if auxl_signal is not None:
            auxl_signal_reshape = auxl_signal.permute(1, 0, 2)

        initial_states = self.init_rnn_state()
        time = feature_input_noise.size(1)

        def compute(i, states, last_output, all_output, gen_flag,
                    all_gen_flag, all_cur_argmax, last_cell_output):
            input_all = [all_discrete_attribute]
            if auxl_signal is not None:
                input_all.append(auxl_signal_reshape[i])
            if self.noise:
                input_all.append(feature_input_noise_reshape[i])
            if self.feedback:
                if feature_input_data_dim == 3:
                    input_all.append(feature_input_data_reshape[i])
                else:
                    input_all.append(last_output)
            input_all = torch.cat(input_all, dim=1)

            new_states = []
            for s, (layer, state) in enumerate(zip(self.rnn_network, states)):
                cell_new_output, cell_new_state = layer(input_all if s == 0
                                                        else cell_new_output, 
                                                        state)
                new_states.append((cell_new_output, cell_new_state))

            new_output_all = []
            id_ = 0
            for j in range(self.sample_len):
                for k in range(len(self.feature_outputs)):
                    output = self.feature_outputs[k]
                    for _, layer in enumerate(self.rnn_output[f"output_{id_}"]):
                        sub_output = layer(cell_new_output)
                    if output.type_.value == OutputType.DISCRETE.value:
                        sub_output = F.softmax(sub_output, dim=1)
                    elif output.type_.value == OutputType.CONTINUOUS.value:
                        if (output.normalization.value ==
                            Normalization.ZERO_ONE.value):
                            sub_output = torch.sigmoid(sub_output)
                        elif (output.normalization.value ==
                              Normalization.MINUSONE_ONE.value):
                            sub_output = torch.tanh(sub_output)
                        else:
                            raise Exception("unknown normalization type")
                    else:
                        raise Exception("unknown output type")
                    new_output_all.append(sub_output)
                    id_ += 1
            new_output = torch.cat(new_output_all, dim=1)
            all_output[i] = new_output

            for j in range(self.sample_len):
                all_gen_flag[i * self.sample_len + j] = gen_flag
                idx = j * len(self.feature_outputs) + self.gen_flag_id
                gen_flag_tensor = new_output_all[idx]
                cur_gen_flag = (torch.argmax
                                (gen_flag_tensor, dim=1) == 0).float().view(-1, 1)
                all_cur_argmax[i * self.sample_len + j] =\
                    torch.argmax(new_output_all
                                 [j * len(self.feature_outputs) +
                                  self.gen_flag_id], dim=1)
                gen_flag *= cur_gen_flag
                gen_flag.detach_()
            return (i + 1, new_states, new_output,
                    all_output, gen_flag, all_gen_flag,
                    all_cur_argmax, cell_new_output)

        i = 0
        states = initial_states
        new_output = (feature_input_data
                      if feature_input_data_dim == 2
                      else feature_input_data_reshape[0])
        feature = torch.zeros((time, self.batch_size,
                               self.feature_out_dim),
                              dtype=torch.float32).to(self.device)
        gen_flag = torch.ones(self.batch_size, 1).to(self.device)
        all_gen_flag = torch.zeros((time * self.sample_len,
                                    self.batch_size, 1),
                                   dtype=torch.float32).to(self.device)
        all_cur_argmax = torch.zeros((time * self.sample_len,
                                      self.batch_size),
                                     dtype=torch.int64).to(self.device)
        cell_output = torch.zeros((self.batch_size,
                                   self.feature_num_units)).to(self.device)

        while i < time and torch.max(gen_flag).item() == 1:
            i, states, _, feature, _, all_gen_flag, all_cur_argmax, _ = compute(
                i, states, new_output, feature, gen_flag,
                all_gen_flag, all_cur_argmax, cell_output)
                   
        def fill_rest(i, all_output, all_gen_flag, all_cur_argmax):
            all_output[i] = torch.zeros((self.batch_size, self.feature_out_dim))
            for j in range(self.sample_len):
                all_gen_flag[i * self.sample_len + j] =\
                    torch.zeros((self.batch_size, 1)).to(self.device)
                all_cur_argmax[i * self.sample_len + j] =\
                    torch.zeros((self.batch_size,), dtype=torch.int64).to(self.device)
            return i + 1, all_output, all_gen_flag, all_cur_argmax

        while i < time:
            _, feature, all_gen_flag, all_cur_argmax = fill_rest(
                i, feature, all_gen_flag, all_cur_argmax)
            i += 1

        all_gen_flag = all_gen_flag.permute(1, 0, 2)
        all_cur_argmax = all_cur_argmax.permute(1, 0)
        length = torch.sum(all_gen_flag, dim=[1, 2])

        feature = feature.permute(1, 0, 2)
        gen_flag_t = all_gen_flag.view(self.batch_size, time, self.sample_len)
        gen_flag_t = torch.sum(gen_flag_t, dim=2)
        gen_flag_t = (gen_flag_t > 0.5).float().unsqueeze(2)
        gen_flag_t = gen_flag_t.repeat(1, 1, self.feature_out_dim)
        feature = feature * gen_flag_t
        feature = feature.reshape(self.batch_size, time * self.sample_len,
                                  self.feature_out_dim // self.sample_len)
        # endregion

        return feature, all_attribute, all_gen_flag, length, all_cur_argmax   
    # endregion
# endregion

