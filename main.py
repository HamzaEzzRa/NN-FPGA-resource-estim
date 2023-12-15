import os
from datetime import datetime

import hls4ml
from data_gen import GeneratorSettings, generate_fc_network
from network_parser import *
from synthesis import to_hls

from utils import *

if __name__ == '__main__':
    # settings = GeneratorSettings(
    #     input_range=Power2Range(16, 128),
    #     layer_range=IntRange(2, 6),
    #     neuron_range=Power2Range(4, 64),
    #     output_range=IntRange(1, 100),
        
    #     parameter_limit=4096,
        
    #     activations=[
    #         None,
    #         'relu',
    #         'softmax',
    #     ],
        
    #     # global_bias_probability=1.0,
    #     # global_bn_probability=0.0,
    #     # global_dropout_probability=0.0,
    #     # global_skip_probability=0.0,
        
    #     verbose=1
    # )
    
    settings = GeneratorSettings(
        input_range=Power2Range(16, 1024),
        layer_range=IntRange(2, 20),
        neuron_range=Power2Range(4, 4096),
        output_range=IntRange(1, 1000),
        
        parameter_limit=4096,
        
        activations=[
            None,
            'relu',
            'sigmoid',
            'tanh',
            'softmax',
        ],
        
        # global_bias_probability=1.0,
        # global_bn_probability=0.0,
        # global_dropout_probability=0.0,
        # global_skip_probability=0.0,
        
        verbose=1
    )
    
    n_models = 100
    while True:
        try:
            reuse_factor = int(np.random.choice([1, 2, 4, 8, 16, 32, 64]))
            precision = np.random.choice([
                'ap_fixed<8, 4>',
                'ap_fixed<8, 3>',
                'ap_fixed<2, 1>',
                'ap_fixed<16, 6>'
            ])
            strategy = np.random.choice(['Latency', 'Resource', 'Resource'])
            board = np.random.choice([
                'pynq-z2',
                'zcu102',
                'alveo-u200'
            ])
            rnd_model = generate_fc_network(settings)
            model_config = parse_keras_config(rnd_model, reuse_factor)
            # print(model_config)
                        
            hls_output_dir = './hls4ml_prj'
            precisions = {
                'Model': precision
            }
            hls_model, hls_config = to_hls(
                rnd_model,
                hls_output_dir,
                precisions,
                strategy=strategy,
                reuse_factor=reuse_factor,
                board=board
            )

            hls_model.build(csim=False, synth=True, export=False, bitfile=False)
            # hls4ml.report.read_vivado_report(hls_output_dir)

            res_report, latency_report = res_from_report(os.path.join(
                hls_output_dir,
                'myproject_prj/solution1/syn/report/myproject_axi_csynth.rpt'
            ))
            save_to_json(
                model_config, hls_config, res_report, latency_report, board
            )
            
            tf.keras.backend.clear_session()
        except Exception as e:
            print(e)
