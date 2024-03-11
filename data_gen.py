from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Input,
)
from keras.models import Model

from utils import *


@dataclass
class GeneratorSettings:
    # Network generation parameters
    input_range: Power2Range = Power2Range(16, 1024)
    layer_range: IntRange = IntRange(2, 20)
    neuron_range: Power2Range = Power2Range(2, 4096)
    output_range: IntRange = IntRange(1, 1000)
    activations: list = field(default_factory = lambda: [
        None,
        'relu',
        'tanh',
        'sigmoid',
        'softmax'
    ])
    
    dropout_range: FloatRange = FloatRange(0.1, 0.8)
    
    # If 0, the number of parameters per layer is not limited
    parameter_limit: int = 4096

    # Layer probabilities
    global_bias_probability: float = 0.9
    global_bn_probability: float = 0.2
    global_dropout_probability: float = 0.4
    global_skip_probability: float = 0.15
    
    bias_probability_func: Callable[[float], float] = lambda x: 0.9
    bn_probability_func: Callable[[float], float] = lambda x: max(0, 0.8 - 2**(-x/5))
    dropout_probability_func: Callable[[float], float] = lambda x: max(0, 0.8 - 2**(-x/5))
    skip_probability_func: Callable[[float], float] = lambda x: max(0, 0.8 - 2**(-x/5))

    # 0: quiet, 1: general info, 2: all info
    verbose: int = 0

def generate_fc_layer(
    inputs,
    units,
    activation=None,
    use_bias=True,
    use_bn=False,
    dropout_rate=0.0,
    skip_inputs=None,
):
    out_list = []
    out_list.append(Dense(units, use_bias=use_bias)(inputs))

    if skip_inputs is not None:
        skip_layer = Add() if out_list[-1].shape[1] == skip_inputs.shape[1]\
            else Concatenate()
        out_list.append(skip_layer([skip_inputs, out_list[-1]]))
    
    if activation is not None:
        out_list.append(Activation(activation)(out_list[-1]))
    
    if use_bn:
        out_list.append(BatchNormalization()(out_list[-1]))
    
    if dropout_rate > 0.0:
        out_list.append(Dropout(dropout_rate)(out_list[-1]))
    
    return out_list

def generate_fc_network(settings: GeneratorSettings, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    input_list = []    
    input_shape = (settings.input_range.random_in_range(rng),)
    input_layer = Input(shape=input_shape)
    input_list.append(input_layer)
    
    model_bias = rng.random() < settings.global_bias_probability
    model_bn = rng.random() < settings.global_bn_probability
    model_dropout = rng.random() < settings.global_dropout_probability
    model_skip = rng.random() < settings.global_skip_probability
    
    n_layers = settings.layer_range.random_in_range(rng)
    for i in range(n_layers):
        x = input_list[-1]
        
        use_bias = model_bias and\
            rng.random() < settings.bias_probability_func(i + 1)
        use_bn = model_bn and i < n_layers - 1 and\
            rng.random() < settings.bn_probability_func(i + 1) and\
            (settings.parameter_limit <= 0 or x.shape[1] <= settings.parameter_limit // 4)
        
        dropout_rate = 0.0
        if model_dropout and not use_bn and i < n_layers - 1\
            and rng.random() < settings.dropout_probability_func(i + 1):
                dropout_rate = settings.dropout_range.random_in_range(rng)
        
        skip_inputs = None
        if model_skip and\
            i > 0 and rng.random() < settings.skip_probability_func(i + 1):
            skip_inputs = input_list[
                rng.integers(0, high=len(input_list) - 1)
            ]
                
        unit_range = settings.neuron_range if i < n_layers - 1\
            else settings.output_range
        range_class_name = unit_range.__class__.__name__
        units = unit_range.random_in_range(rng)
        if settings.parameter_limit > 0:
            max_units = min(units, settings.parameter_limit // x.shape[1])
            if use_bias:
                bias_limit = (settings.parameter_limit - max_units) // x.shape[1]
                if bias_limit <= 0:
                    max_units = 1
                else:
                    if range_class_name == Power2Range.__name__:
                        bias_limit = 2 ** int(np.log2(bias_limit))
                    max_units = min(
                        max_units,
                        bias_limit
                    )
            if use_bn:
                bn_limit = (settings.parameter_limit // 4) // x.shape[1]
                if bn_limit <= 0:
                    max_units = 1
                else:
                    if range_class_name == Power2Range.__name__:
                        bn_limit = 2 ** int(np.log2(bn_limit))
                    max_units = min(
                        max_units,
                        bn_limit
                    )
            if max_units != units:
                range_class = globals()[unit_range.__class__.__name__]
                units = range_class(settings.neuron_range.min, max_units)\
                    .random_in_range(rng)
        
        activation = rng.choice(settings.activations)
        input_list += generate_fc_layer(
            x,
            units,
            activation,
            use_bias,
            use_bn,
            dropout_rate,
            skip_inputs
        )
    
    model = Model(inputs=input_layer, outputs=input_list[-1])
    model.build([None] + list(input_shape))
    
    if settings.verbose > 0:
        model.summary()
    
    return model

def get_submodels(base_model, verbose=0):
    submodels = []
    sublayers = []
    
    base_input = base_model.input
    input_shape = base_model.input_shape
    for layer in base_model.layers[1:]:
        sublayers.append(layer.output)

        model = Model(inputs=base_input, outputs=sublayers[-1])
        model.build([None] + list(input_shape))

        if verbose > 0:
            model.summary()
        
        submodels.append(model)
    
    return submodels

if __name__ == '__main__':
    rng = np.random.default_rng(1337)
    nn = generate_fc_network(GeneratorSettings(), rng)

    get_submodels(nn, verbose=1)
