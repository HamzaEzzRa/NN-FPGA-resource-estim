import json
import os
from glob import glob

import numpy as np

from utils import get_closest_reuse_factor

layer_type_map = {
    'inputlayer': 1,
    'dense': 2,
    'relu': 3,
    'sigmoid': 4,
    'tanh': 5,
    'softmax': 6,
    'batchnormalization': 7,
    'add': 8,
    'concatenate': 9,
    'dropout': 10,
}

precision_map = {
    'ap_fixed<2, 1>': 1,
    'ap_fixed<8, 3>': 2,
    'ap_fixed<8, 4>': 3,
    'ap_fixed<16, 6>': 4,
}

strategy_map = {
    'latency': 1,
    'resource': 2
}

boards_file = './supported_boards.json'
boards_data = {}
with open(boards_file, 'r') as json_file:
    boards_data = json.load(json_file)

board_map = {
    key.lower(): value + 1 for value, key in enumerate(boards_data.keys())
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

def padded_data_from_json(name_pattern):
    json_data = []
    json_files = glob(name_pattern)
    for filename in json_files:
        with open(filename, 'r') as json_file:
            json_data += json.load(json_file)

    inputs = []
    targets = []
    max_layer_depth = 0
    for model in json_data:
        max_layer_depth = max(max_layer_depth, len(model['model_config']))
    
    for model in json_data:
        config_data = []
        
        hls_config = model['hls_config']['Model']
        hls_precision = hls_config['Precision']
        hls_strategy = hls_config['Strategy']
        
        target_board = model['board']
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
            reuse_factor = layer['reuse_factor']
            
            config_data.append([
                layer_type_map[layer_type.lower()],
                strategy_map[hls_strategy.lower()],
                precision_map[hls_precision.lower()],
                board_map[target_board.lower()],
                input_shape,
                output_shape,
                layer_parameters,
                reuse_factor,
            ])
        
        for i in range(max_layer_depth - len(config_data)):
            config_data.append([0] * len(config_data[0]))
        
        inputs.append(config_data)

        res_report = model['res_report']
        bram = res_report['BRAM']
        dsp = res_report['DSP']
        ff = res_report['FF']
        lut = res_report['LUT']
        
        latency_report = model['latency_report']
        cycles_min = latency_report['cycles_min']
        cycles_max = latency_report['cycles_max']
        # estimated_clock = latency_report['estimated_clock']
        
        board_data = boards_data['pynq-z2']
        # board_data = boards_data[target_board]
        max_bram = board_data['max_bram']
        max_dsp = board_data['max_dsp']
        max_ff = board_data['max_ff']
        max_lut = board_data['max_lut']
        
        targets.append([
            bram / max_bram,
            dsp / max_dsp,
            ff / max_ff,
            lut / max_lut,
            # cycles_min,
            # cycles_max,
            # estimated_clock
        ])

    inputs = np.asarray(inputs)
    targets = np.asarray(targets)

    return inputs, targets

def split_data_from_json(name_pattern):
    pass

if __name__ == '__main__':
    inputs, targets = padded_data_from_json('./datasets/*.json')
    print(len(inputs))
    
    rnd_idx = np.random.randint(0, high=len(inputs))
    print(inputs[rnd_idx])
    print(targets[rnd_idx])
