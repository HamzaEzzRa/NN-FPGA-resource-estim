import json
import os

import numpy as np

from utils import get_closest_reuse_factor

layer_type_map = {
    'inputlayer': 1,
    'dense': 2,
    'relu': 3,
    'softmax': 4,
    'batchnormalization': 5,
    'add': 6,
    'concatenate': 7,
    'dropout': 8,
}

def parse_keras_config(model, rf):
    layers_data = []
    for layer in model.layers:
        class_name = layer.__class__.__name__
        layer_config = layer.get_config()
        layer_weights = layer.get_weights()
        
        layer_dict = {}
        layer_dict['class_name'] = class_name
        
        input_shape = layer.input_shape
        if isinstance(input_shape, list) and len(input_shape) == 1:
            input_shape = input_shape[0]
        layer_dict['input_shape'] = input_shape
        
        output_shape = layer.output_shape
        if isinstance(output_shape, list) and len(output_shape) == 1:
            output_shape = output_shape[0]
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
            layer_dict['use_bias'] = layer_config['use_bias']
        elif class_name == 'Activation':
            layer_dict['activation'] = layer_config['activation']
        
        layer_dict['dtype'] = layer_config['dtype']
        
        layer_dict['reuse_factor'] = rf
        if class_name in ['Dense']:
            layer_dict['reuse_factor'] = get_closest_reuse_factor(
                np.prod([
                    x for x in input_shape if x is not None
                ]),
                np.prod([
                    x for x in output_shape if x is not None
                ]),
                rf
            )
        
        layers_data.append(layer_dict)
    
    return layers_data

def save_to_json(
    model_config,
    hls_config,
    res_report,
    latency_report,
    board,
    file_path='./dataset.json',
    indent=2
):
    model_info = {
        'model_config': model_config,
        'hls_config': hls_config,
        'res_report': res_report,
        'latency_report': latency_report,
        'board': board
    }
    
    try:
        with open(file_path, 'r+') as json_file:
            json_file.seek(os.stat(file_path).st_size - 2)
            json_file.write(',\n' + ' ' * indent + '{}\n]'.format(
                json.dumps(model_info, indent=indent)
            ))
    except (FileNotFoundError, json.JSONDecodeError):
        with open(file_path, 'w') as json_file:
            json.dump([model_info], json_file, indent=indent)
    
def load_from_json(
    file_path
):
    json_data = []
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)

    inputs = []
    targets = []
    max_layer_depth = 0
    for model in json_data:
        max_layer_depth = max(max_layer_depth, len(model['model_config']))
    
    for model in json_data:
        config_data = []
        model_config = model['model_config']
        for layer in model_config:
            layer_type = layer['class_name']
            if layer_type == 'Activation':
                layer_type = layer['activation']
            
            input_shape = layer['input_shape']
            if layer_type in ['Add', 'Concatenate']:
                input_shape = layer['output_shape']
            input_shape = np.prod([
                x for x in input_shape if x is not None
            ])
            
            output_shape = np.prod([
                x for x in layer['output_shape'] if x is not None
            ])
            layer_parameters = layer['parameters']
            # reuse_factor = layer['reuse_factor']
            
            config_data.append([
                layer_type_map[layer_type.lower()],
                input_shape,
                output_shape,
                layer_parameters,
                # reuse_factor
            ])
        
        for i in range(max_layer_depth - len(config_data)):
            config_data.append([0] * len(config_data[0]))
        inputs.append(config_data)

        hls_report = model['hls_report']
        bram = hls_report['BRAM']
        dsp = hls_report['DSP']
        ff = hls_report['FF']
        lut = hls_report['LUT']
        targets.append([
            bram / 280.,
            dsp / 220.,
            ff / 106400.,
            lut / 53200.
        ])

    return np.asarray(inputs), np.asarray(targets)
