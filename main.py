import os
from datetime import datetime

import hls4ml
from data_gen import GeneratorSettings, generate_fc_network
from network_parser import *
from synthesis import to_hls

from utils import *

if __name__ == '__main__':
    settings = GeneratorSettings(
        input_range=Power2Range(16, 128),
        layer_range=IntRange(2, 6),
        neuron_range=Power2Range(4, 64),
        output_range=IntRange(1, 100),
        
        parameter_limit=4096,
        
        activations=[
            None,
            'relu',
            'softmax',
        ],
        
        # global_bias_probability=1.0,
        # global_bn_probability=1.0,
        # global_dropout_probability=1.0,
        # global_skip_probability=1.0,
        
        verbose=1
    )
    
    n_models = 1000
    for i in range(n_models):
        try:
            rnd_model = generate_fc_network(settings)
            model_config = parse_keras_config(rnd_model)
            # print(model_config)
                        
            hls_output_dir = './hls4ml_prj'
            precisions = {
                'Model': 'ap_fixed<16, 6>'
            }
            hls_model, hls_config = to_hls(
                rnd_model,
                hls_output_dir,
                precisions,
                strategy='Resource',
                reuse_factor=32
            )

            hls_model.build(csim=False, synth=True, export=False, bitfile=False)
            # hls4ml.report.read_vivado_report(hls_output_dir)

            hls_report = res_from_report(os.path.join(
                hls_output_dir,
                'myproject_prj/solution1/syn/report/myproject_axi_csynth.rpt'
            ))
            save_to_json(model_config, hls_config, hls_report)
        except Exception as e:
            print(e)
