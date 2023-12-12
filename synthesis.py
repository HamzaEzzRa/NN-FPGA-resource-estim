import os

import hls4ml

os.environ['PATH'] = '/opt/Xilinx/Vivado/2019.1/bin:' + os.environ['PATH']

def print_hls_config(d, indent=0):
    align = 20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_hls_config(value, indent + 1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

def to_hls(model, output_dir, precisions, strategy, reuse_factor):
    config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    # config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Precision'] = precisions['Model']
    config['Model']['Strategy'] = strategy
    config['Model']['ReuseFactor'] = reuse_factor

    # for idx, layer in enumerate(config['LayerName'].keys()):
    #     if idx >= len(precisions['Layers']):
    #         break
    #     config['LayerName'][layer]['Precision'] = precisions['Layers'][idx]

    print("-----------------------------------")
    print("Configuration")
    print_hls_config(config)
    print("-----------------------------------")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend='VivadoAccelerator',
        clock_period='10',
        board='pynq-z2'
    )
    hls_model.compile()

    return hls_model, config
