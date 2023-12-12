import json
import os

import numpy as np


def parse_keras_config(model):
    layers_data = []
    for layer in model.layers:
        class_name = layer.__class__.__name__
        layer_config = layer.get_config()
        layer_weights = layer.get_weights()
        
        layer_dict = {}
        layer_dict['class_name'] = class_name
        
        input_shape = layer.input_shape
        layer_dict['input_shape'] = input_shape
        
        output_shape = layer.output_shape
        layer_dict['output_shape'] = output_shape
        
        parameter_count = 0
        for weight_group in layer_weights:
            parameter_count += np.size(weight_group)
        layer_dict['parameters'] = parameter_count
        
        trainable_parameter_count = 0
        for var_group in layer.trainable_variables:
            trainable_parameter_count += np.size(var_group)
        layer_dict['trainable_parameters'] = trainable_parameter_count
        
        if class_name == 'Dense':
            layer_dict['neurons'] = int(layer_config['units'])
        elif class_name == 'Activation':
            layer_dict['activation'] = layer_config['activation']
        elif class_name == 'BatchNormalization':
            pass
        elif class_name == 'Add':
            pass
        elif class_name == 'Concatenate':
            pass
        layer_dict['dtype'] = layer_config['dtype']
        
        layers_data.append(layer_dict)
    
    return layers_data

def save_to_json(
    model_config,
    hls_config,
    hls_report,
    save_file='./dataset.json'
):
    try:
        with open(save_file, 'r') as json_file:
            models_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        models_info = []

    models_info.append({
        'model_config': model_config,
        'hls_config': hls_config,
        'hls_report': hls_report
    })

    with open(save_file, 'w') as json_file:
        json.dump(models_info, json_file, indent=2)

def load_from_json():
    pass
