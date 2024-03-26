import json
import os
from dataclasses import asdict, dataclass, field
from glob import glob
from itertools import permutations

import numpy as np
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Input
from keras.models import Model
from tqdm import tqdm

from utils import get_closest_reuse_factor, limited_reservoir_sampling, rnd_permutations


@dataclass
class NetworkDataFilter:
    # The min/max settings are ignored if <= 0
    min_layers: int = 0
    max_layers: int = 0
    min_reuse_factor: int = 0
    max_reuse_factor: int = 0
    
    min_input_size: int = 0
    max_input_size: int = 0
    
    min_output_size: int = 0
    max_output_size: int = 0
    
    min_softmax_count: int = 0
    max_softmax_count: int = 0
    
    # Exclude models that contains specific layers
    exclude_layers: list = field(default_factory = lambda: [])
    
    strategies: list = field(default_factory = lambda: [
        'Resource',
        'Latency'
    ])
    
    precisions: list = field(default_factory = lambda: [
        'ap_fixed<2, 1>',
        'ap_fixed<8, 3>',
        'ap_fixed<8, 4>',
        'ap_fixed<16, 6>'
    ])
    
    boards: list = field(default_factory = lambda: [
        'pynq-z2',
        'zcu102',
        'alveo-u200'
    ])

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

def filter_match(model_data, data_filter: NetworkDataFilter):
    n_layers = len(model_data['model_config'])
    if (data_filter.min_layers > 0 and n_layers < data_filter.min_layers)\
    or (data_filter.max_layers > 0 and n_layers > data_filter.max_layers):
        return False
    
    input_size = np.prod([x for x in model_data['model_config'][0]['input_shape'] if x is not None])
    if (data_filter.min_input_size > 0 and input_size < data_filter.min_input_size)\
    or (data_filter.max_input_size > 0 and input_size > data_filter.max_input_size):
        return False
    
    output_size = np.prod([x for x in model_data['model_config'][-1]['output_shape'] if x is not None])
    if (data_filter.min_output_size > 0 and output_size < data_filter.min_output_size)\
    or (data_filter.max_output_size > 0 and output_size > data_filter.max_output_size):
        return False
    
    softmax_count = 0
    for layer_data in model_data['model_config']:
        reuse_factor = layer_data['reuse_factor']
        if (data_filter.min_reuse_factor > 0 and reuse_factor < data_filter.min_reuse_factor)\
        or (data_filter.max_reuse_factor > 0 and reuse_factor > data_filter.max_reuse_factor):
            return False

        layer_class = layer_data['class_name']
        if layer_class in data_filter.exclude_layers:
            return False
    
        if layer_class == 'Activation':
            if layer_data['activation'].lower() == 'softmax':
                softmax_count += 1
            if layer_data['activation'] in data_filter.exclude_layers:
                return False

    if (data_filter.min_softmax_count > 0 and softmax_count < data_filter.min_softmax_count)\
    or (data_filter.max_softmax_count > 0 and softmax_count > data_filter.max_softmax_count):
        return False
    
    hls_data = model_data['hls_config']['Model']
    strategy = hls_data['Strategy']
    if strategy not in data_filter.strategies:
        return False
    
    precision = hls_data['Precision']
    if precision not in data_filter.precisions:
        return False
    
    board = model_data['board']
    if board not in data_filter.boards:
        return False

    return True

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

def dump_json(models_info, file_path='./dataset.json', indent=2):
    with open(file_path, 'w') as json_file:
        json.dump([models_info[0]], json_file, indent=indent)
        for model_info in models_info[1:]:
            json_file.seek(os.stat(file_path).st_size - 2)
            json_file.write(',\n' + ' ' * indent + '{}\n]'.format(
                json.dumps(model_info, indent=indent)
            ))

def simple_data_from_json(name_pattern, data_filter: NetworkDataFilter = None):
    json_data = []
    json_files = glob(name_pattern)
    for filename in json_files:
        with open(filename, 'r') as json_file:
            json_data += json.load(json_file)

    # Add filtering to the json data
    if data_filter is not None:
        filtered_data = []
        for model_data in json_data:
            if filter_match(model_data, data_filter):
                filtered_data.append(model_data)

        json_data = filtered_data

    inputs = []
    targets = []
    for model_data in json_data:
        target_board = model_data['board']
        model_config = model_data['model_config']
        
        input_layer_count = len([x for x in model_config if x['class_name'] == 'InputLayer'])
        dense_layer_count = len([x for x in model_config if x['class_name'] == 'Dense'])
        bn_layer_count = len([x for x in model_config if x['class_name'] == 'BatchNormalization'])
        add_layer_count = len([x for x in model_config if x['class_name'] == 'Add'])
        concatenate_layer_count = len([x for x in model_config if x['class_name'] == 'Concatenate'])
        dropout_layer_count = len([x for x in model_config if x['class_name'] == 'Dropout'])
        
        relu_layer_count = len(
            [x for x in model_config\
            if x['class_name'] == 'Activation' and x['activation'] == 'relu']
        )
        sigmoid_layer_count = len(
            [x for x in model_config\
            if x['class_name'] == 'Activation' and x['activation'] == 'sigmoid']
        )
        tanh_layer_count = len(
            [x for x in model_config\
            if x['class_name'] == 'Activation' and x['activation'] == 'tanh']
        )
        softmax_layer_count = len(
            [x for x in model_config\
            if x['class_name'] == 'Activation' and x['activation'] == 'softmax']
        )
        
        avg_dense_parameters = np.mean([x['parameters'] for x in model_config if x['class_name'] == 'Dense'])
        avg_dense_inputs = np.mean([np.prod([x for x in layer['input_shape'] if x is not None]) for layer in model_config if layer['class_name'] == 'Dense'])
        avg_dense_outputs = np.mean([np.prod([x for x in layer['output_shape'] if x is not None]) for layer in model_config if layer['class_name'] == 'Dense'])
        avg_dense_reuse_factor = np.mean([x['reuse_factor'] for x in model_config if x['class_name'] == 'Dense'])
        
        hls_config = model_data['hls_config']['Model']
        hls_precision = hls_config['Precision']
        hls_strategy = hls_config['Strategy']
        
        info = []
        if 'model_id' in model_data:
            info.append(model_data['model_id'])
        inputs.append(info + [
            hls_strategy.lower(),
            hls_precision.lower(),
            target_board.lower(),
            # strategy_map[hls_strategy.lower()],
            # precision_map[hls_precision.lower()],
            # board_map[target_board.lower()],
            # input_layer_count,
            dense_layer_count,
            bn_layer_count,
            add_layer_count,
            concatenate_layer_count,
            dropout_layer_count,
            relu_layer_count,
            sigmoid_layer_count,
            tanh_layer_count,
            softmax_layer_count,
            avg_dense_parameters,
            avg_dense_inputs,
            avg_dense_outputs,
            avg_dense_reuse_factor
        ])
        
        res_report = model_data['res_report']
        bram = res_report['BRAM']
        dsp = res_report['DSP']
        ff = res_report['FF']
        lut = res_report['LUT']
        
        # latency_report = model_data['latency_report']
        # cycles_min = latency_report['cycles_min']
        # cycles_max = latency_report['cycles_max']
        # estimated_clock = latency_report['estimated_clock']
        
        # board_data = boards_data['zcu102']
        board_data = boards_data[target_board]
        max_bram = board_data['max_bram']
        max_dsp = board_data['max_dsp']
        max_ff = board_data['max_ff']
        max_lut = board_data['max_lut']
        
        targets.append([
            # max(1 / max_bram, min(bram / max_bram, 2.0)) * 100,
            # max(1 / max_dsp, min(dsp / max_dsp, 2.0)) * 100,
            # max(1 / max_ff, min(ff / max_ff, 2.0)) * 100,
            # max(1 / max_lut, min(lut / max_lut, 10.0)) * 100,
            # min(bram / max_bram, 2.0) * 100,
            # min(dsp / max_dsp, 2.0) * 100,
            # min(ff / max_ff, 2.0) * 100,
            # min(lut / max_lut, 2.0) * 100,
            (bram / max_bram) * 100,
            (dsp / max_dsp) * 100,
            (ff / max_ff) * 100,
            (lut / max_lut) * 100
        ])

    inputs = np.asarray(inputs)
    targets = np.asarray(targets, dtype=np.float64)
    
    return inputs, targets

def parse_layer_data(layer_config):
    layer_type = layer_config['class_name']
    if layer_type == 'Activation':
        layer_type = layer_config['activation']
    
    input_shape = layer_config['input_shape']
    if layer_type in ['Add', 'Concatenate']:
        input_shape = layer_config['output_shape']
    input_shape = np.prod([
        x for x in input_shape if x is not None
    ])
    
    output_shape = np.prod([
        x for x in layer_config['output_shape'] if x is not None
    ])
    layer_parameters = layer_config['parameters']
    reuse_factor = layer_config['reuse_factor']

    return [
        layer_type_map[layer_type.lower()],
        input_shape,
        output_shape,
        layer_parameters,
        reuse_factor,
    ]

def padded_data_from_json(name_pattern, data_filter: NetworkDataFilter = None):
    json_data = []
    json_files = glob(name_pattern)
    for filename in json_files:
        with open(filename, 'r') as json_file:
            json_data += json.load(json_file)

    # Add filtering to the json data
    if data_filter is not None:
        filtered_data = []
        for model in json_data:
            if filter_match(model, data_filter):
                filtered_data.append(model)

        json_data = filtered_data

    global_inputs = []
    seq_inputs = []
    targets = []
    max_layer_depth = 0
    for model in json_data:
        max_layer_depth = max(max_layer_depth, len(model['model_config']))
    
    for model in json_data:
        target_board = model['board']
        
        layers_data = []
        
        hls_config = model['hls_config']['Model']
        hls_precision = hls_config['Precision']
        hls_strategy = hls_config['Strategy']
        
        model_config = model['model_config']
        for layer_config in model_config:
            layers_data.append(
                [precision_map[hls_precision.lower()]] + parse_layer_data(layer_config)
            )
                
        for i in range(max_layer_depth - len(layers_data)):
            layers_data.append([0] * len(layers_data[0]))
        
        global_inputs.append([
            strategy_map[hls_strategy.lower()],
            board_map[target_board.lower()],
            len(model_config)
        ])
        
        seq_inputs.append(layers_data)

        res_report = model['res_report']
        bram = res_report['BRAM']
        dsp = res_report['DSP']
        ff = res_report['FF']
        lut = res_report['LUT']
        
        latency_report = model['latency_report']
        cycles_min = latency_report['cycles_min']
        cycles_max = latency_report['cycles_max']
        # estimated_clock = latency_report['estimated_clock']
        
        # board_data = boards_data['zcu102']
        board_data = boards_data[target_board]
        max_bram = board_data['max_bram']
        max_dsp = board_data['max_dsp']
        max_ff = board_data['max_ff']
        max_lut = board_data['max_lut']
        
        # targets.append([
        #     max(1, bram) / max_bram,
        #     # max(1, dsp) / max_dsp,
        #     # max(1, ff) / max_ff,
        #     # max(1, lut) / max_lut,
        #     # cycles_min,
        #     # cycles_max,
        #     # estimated_clock
        # ])
        
        # targets.append([
        #     max(0, min(bram, max_bram + 1)),
        #     max(0, min(dsp, max_dsp + 1)),
        #     max(0, min(ff, max_ff + 1)),
        #     max(0, min(lut, max_lut + 1)),
        #     # cycles_min,
        #     # cycles_max,
        #     # estimated_clock
        # ])
        
        # targets.append([
        #     max(1, min(bram, 2.0 * max_bram)),
        #     max(1, min(dsp, 2.0 * max_dsp)),
        #     max(1, min(ff, 2.0 * max_ff)),
        #     max(1, min(lut, 2.0 * max_lut)),
        #     # cycles_min,
        #     # cycles_max,
        #     # estimated_clock
        # ])
        
        targets.append([
            # max(1 / max_bram, min(bram / max_bram, 2.0)) * 100,
            # max(1 / max_dsp, min(dsp / max_dsp, 2.0)) * 100,
            # max(1 / max_ff, min(ff / max_ff, 2.0)) * 100,
            max(1 / max_lut, min(lut / max_lut, 2.0)) * 100,
        #     # cycles_min,
        #     # cycles_max,
        #     # estimated_clock
        ])

    global_inputs = np.asarray(global_inputs)
    seq_inputs = np.asarray(seq_inputs)
    targets = np.asarray(targets, dtype=np.float64)

    # targets -= targets.mean()
    # targets /= targets.std()

    # min_vals = np.min(targets, axis=0)
    # max_vals = np.max(targets, axis=0)
    # targets = (targets - min_vals) / (max_vals - min_vals)
    
    # print(targets)

    return global_inputs, seq_inputs, targets

def split_data_from_json(name_pattern):
    pass

def sanitize_json_data(json_data, save_path=None, indent=2):
    for idx in range(len(json_data)):
        json_data[idx]['model_id'] = idx
    
    if save_path is not None and save_path != '':
        with open(save_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=indent)

    return json_data

def sanitize_json_files(name_pattern, save_path=None, indent=2):
    json_data = []
    json_files = glob(name_pattern)
    for filename in json_files:
        with open(filename, 'r') as json_file:
            json_data += json.load(json_file)
    
    return sanitize_json_data(json_data, save_path)

def layer_from_json(layer_info):
    layer = None
    if layer_info['class_name'] == 'Dense':
        layer = Dense(layer_info['neurons'], use_bias=layer_info['use_bias'])
    elif layer_info['class_name'] == 'Activation':
        layer = Activation(layer_info['activation'])
    elif layer_info['class_name'] == 'BatchNormalization':
        layer = BatchNormalization()
    elif layer_info['class_name'] == 'Dropout':
        layer = Dropout(0.25)

    return layer

def network_from_json(model_json):
    input_shape = model_json[0]['input_shape']
    inputs = Input(shape=input_shape[1:])
    
    x = inputs
    for layer_info in model_json[1:]:
        x = layer_from_json(layer_info)(x)
    outputs = x

    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_model_permutations(model_json, max_permutations):
    # keras_model = network_from_json(model_json)
    layer_count = len(model_json)
    
    last_dense_idx = 0
    for idx, layer_info in enumerate(model_json[::-1]):
        if layer_info['class_name'] == 'Dense':
            last_dense_idx = layer_count - idx - 1
            break
    
    # import time
    # start_time = time.time()
    # Reservoir sampling
    # permut_layer_indices = limited_reservoir_sampling(
    #     permutations(list(range(1, last_dense_idx))),
    #     max_permutations,
    #     max_iters=100000
    # )
    permut_layer_indices = rnd_permutations(
        range(1, last_dense_idx),
        max_permutations
    )
    # print(f'Permutation Time: {time.time() - start_time}')
    
    # print(permut_layer_indices)
    input_shape = model_json[0]['input_shape']
    # print(input_shape)
    inputs = Input(shape=input_shape[1:])
    permut_models = []
    # start_time = time.time()
    for layer_indices in permut_layer_indices:
        x = inputs
        for idx in layer_indices:
            layer_info = model_json[idx]
            x = layer_from_json(layer_info)(x)
        for layer_info in model_json[last_dense_idx:]:
            x = layer_from_json(layer_info)(x)
        outputs = x
        
        model = Model(inputs=inputs, outputs=outputs)        
        model.build(input_shape)
        
        permut_models.append(model)
    # print(f'Build Time: {time.time() - start_time}')
    
    return permut_models

def augment_dataset(name_pattern, max_permutations=5, data_filter: NetworkDataFilter = None, save_path=None, indent=2):
    json_data = []
    json_files = glob(name_pattern)
    for filename in json_files:
        with open(filename, 'r') as json_file:
            json_data += json.load(json_file)
    
    if data_filter is not None:
        filtered_data = []
        for model_json in json_data:
            if filter_match(model_json, data_filter):
                filtered_data.append(model_json)

        json_data = filtered_data
    
    for model_json in tqdm(json_data):
        model_config = model_json['model_config']
        hls_config = model_json['hls_config']
        res_report = model_json['res_report']
        latency_report = model_json['latency_report']
        board = model_json['board']
        
        # Save original config
        save_to_json(
            model_config,
            hls_config,
            res_report,
            latency_report,
            board,
            file_path=save_path
        )
        
        # Add in the permutations
        rf = hls_config['Model']['ReuseFactor']
        model_permuts = get_model_permutations(model_config, max_permutations)
        for permut in model_permuts:
            permut_config = parse_keras_config(permut, rf)
            if save_path is not None and save_path != '':
                save_to_json(
                    permut_config,
                    hls_config,
                    res_report,
                    latency_report,
                    board,
                    file_path=save_path
                )

if __name__ == '__main__':
    filename = './datasets/complex/sanitized_train.json'
    
    data_filter = NetworkDataFilter(
        exclude_layers=['Concatenate', 'Add'],
        max_output_size=200,
    )
    global_inputs, seq_inputs, targets = padded_data_from_json(
        filename,
        data_filter
    )
    print(global_inputs.shape)
    print(seq_inputs.shape)
    print(targets.shape)
    
    with open(filename, 'r') as json_file:
        json_data = json.load(json_file)
    
    # model_json = json_data[0]['model_config']
    # permut_models = get_model_permutations(model_json)
    # for model in permut_models:
    #     print('=' * 100)
    #     model.summary()
    
    for perm in [2, 3, 8, 10]:
        augmented_filename = f'./datasets/complex/augmented_train_{perm}p.json'
        augment_dataset(
            filename,
            max_permutations=5,
            data_filter=data_filter,
            save_path=augmented_filename,
            indent=2
        )
        global_inputs, seq_inputs, targets = padded_data_from_json(
            augmented_filename,
            data_filter
        )
        print(global_inputs.shape)
        print(seq_inputs.shape)
        print(targets.shape)
    
    # sanitize_json_files('./datasets/complex/dataset-*.json', './datasets/complex/sanitized_train.json')
    # sanitize_json_files('./datasets/complex/test_dataset.json', './datasets/complex/sanitized_test.json')
