# "model_config": [
#       {
#         "class_name": "InputLayer",
#         "input_shape": [
#           null,
#           32
#         ],
#         "output_shape": [
#           null,
#           32
#         ],
#         "parameters": 0,
#         "trainable_parameters": 0,
#         "dtype": "float32",
#         "reuse_factor": 16
#       },
#       {
#         "class_name": "Dense",
#         "input_shape": [
#           null,
#           32
#         ],
#         "output_shape": [
#           null,
#           4
#         ],
#         "parameters": 132,
#         "trainable_parameters": 132,
#         "neurons": 4,
#         "use_bias": true,
#         "dtype": "float32",
#         "reuse_factor": 16
#       },
#       {
#         "class_name": "Activation",
#         "input_shape": [
#           null,
#           4
#         ],
#         "output_shape": [
#           null,
#           4
#         ],
#         "parameters": 0,
#         "trainable_parameters": 0,
#         "activation": "tanh",
#         "dtype": "float32",
#         "reuse_factor": 16
#       },
#       {
#         "class_name": "Dense",
#         "input_shape": [
#           null,
#           4
#         ],
#         "output_shape": [
#           null,
#           642
#         ],
#         "parameters": 3210,
#         "trainable_parameters": 3210,
#         "neurons": 642,
#         "use_bias": true,
#         "dtype": "float32",
#         "reuse_factor": 12
#       },
#       {
#         "class_name": "Activation",
#         "input_shape": [
#           null,
#           642
#         ],
#         "output_shape": [
#           null,
#           642
#         ],
#         "parameters": 0,
#         "trainable_parameters": 0,
#         "activation": "softmax",
#         "dtype": "float32",
#         "reuse_factor": 16
#       }
#     ],

if __name__ == '__main__':
    # from keras.layers import Activation, Dense, Input
    # from keras.models import Model
    import json

    import matplotlib.pyplot as plt
    import numpy as np

    # input_shape = (32,)
    # inputs = Input(shape=input_shape)
    # x = Dense(4, use_bias=True)(inputs)
    # x = Activation('tanh')(x)
    # x = Dense(642, use_bias=True)(x)
    # x = Activation('softmax')(x)
    # outputs = x
    # model = Model(inputs=inputs, outputs=outputs)
    # model.build([None] + list(input_shape))
    # model.summary()

    filename = './softmax_results.json'
    with open(filename, 'r') as json_file:
        json_data = json.load(json_file)

    n_samples = len(json_data)
    output_sizes = np.linspace(3, 600, num=100, dtype=int)
    keys = ['BRAM_18K', 'DSP', 'FF', 'LUT']
    data = np.asarray([[int(x[key]) for key in keys] for x in json_data]).T
    
    data_len = len(data)
    n = int(np.ceil(np.sqrt(data_len)))
    fig, axis = plt.subplots(n, n, sharex=True)
    axis = np.reshape(axis, -1)
    axis_dict = {
        key: val for key, val in zip(keys, axis[:len(keys)])
    }
    
    get_color = plt.cm.get_cmap('hsv', data_len + 1)
    for idx, res in enumerate(data):
        axis[idx].scatter(output_sizes[:n_samples], res, color=get_color(idx))
        axis[idx].set_title(f'{keys[idx].upper()} Utilization')
        axis[idx].set_xlabel('Softmax Output')
        axis[idx].set_ylabel('# Used')

    # axis_dict['BRAM_18K'].hlines(4320, xmin=output_sizes[0], xmax=output_sizes[-1], color='red')
    # axis_dict['DSP'].hlines(6840, xmin=output_sizes[0], xmax=output_sizes[-1], color='red')
    # axis_dict['FF'].hlines(2364480, xmin=output_sizes[0], xmax=output_sizes[-1], color='red')
    # axis_dict['LUT'].hlines(1182240, xmin=output_sizes[0], xmax=output_sizes[-1], color='red')
    
    plt.show()
