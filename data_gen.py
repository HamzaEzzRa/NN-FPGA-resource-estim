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
    
    n_layers = settings.layer_range.random_in_range(rng)
    for i in range(n_layers):
        use_bias = rng.random() < settings.bias_probability_func(i + 1)
        use_bn = i < n_layers - 1 and\
            rng.random() < settings.bn_probability_func(i + 1)
        
        dropout_rate = 0.0
        if not use_bn and i < n_layers - 1\
            and rng.random() < settings.dropout_probability_func(i + 1):
                dropout_rate = settings.dropout_range.random_in_range(rng)
        
        skip_inputs = None
        if i > 0 and rng.random() < settings.skip_probability_func(i + 1):
            skip_inputs = input_list[rng.integers(0, high=len(input_list) - 1)]
        
        x = input_list[-1]
        
        unit_range = settings.neuron_range if i < n_layers - 1\
            else settings.output_range
        units = unit_range.random_in_range(rng)
        if settings.parameter_limit > 0:
            max_units = units
            if use_bias:
                max_units = min(
                    max_units,
                    2 ** int(np.log2(settings.parameter_limit - max_units))
                )
            if use_bn:
                max_units = min(
                    max_units,
                    2 ** int(np.log2(settings.parameter_limit // 4))
                )
            if max_units != units:
                units = Power2Range(settings.neuron_range.min, max_units)\
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
