import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    Layer,
    LayerNormalization,
    Masking,
    MultiHeadAttention,
)
from tensorflow.keras.losses import huber
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model

from network_parser import *
from synthesis import to_hls

seed_num = 1337
np.random.seed(seed_num)
tf.keras.utils.set_random_seed(seed_num)
tf.config.experimental.enable_op_determinism()

@dataclass
class DenseSettings:
    numerical_dense_layers: list = field(default_factory = lambda: [
        16
    ])
    
    dense_layers: list = field(default_factory = lambda: [
        64,
        32,
        64,
        32
    ])
    
    dense_activations: list = field(default_factory = lambda: [
        'tanh',
        'tanh',
        'relu',
        'relu'
    ])
    
    dense_dropouts: list = field(default_factory = lambda: [
        0.0,
        0.2,
        0.2,
        0.0,
    ])

@dataclass
class LSTMSettings:
    global_dense_layers: list = field(default_factory = lambda: [
        64,
        128
    ])
    seq_dense_layers: list = field(default_factory = lambda: [
        64,
        128
    ])
    
    global_numerical_dense_layers: list = field(default_factory = lambda: [
        32
    ])
    layer_numerical_dense_layers: list = field(default_factory = lambda: [
        32
    ])
    
    lstm_layers: list = field(default_factory = lambda: [
        128,
        32
    ])
    
    embedding_outputs: list = field(default_factory = lambda: [
        16,
        16,
        16,
        16
    ])
    
    dense_layers: list = field(default_factory = lambda: [
        128,
        128,
        64
    ])
    
    dense_dropouts: list = field(default_factory = lambda: [
        0.2,
        0.2,
    ])
    
@dataclass
class TransformerSettings:
    global_dense_layers: list = field(default_factory = lambda: [
        64,
        128
    ])
    seq_dense_layers: list = field(default_factory = lambda: [
        64,
        128
    ])
    
    global_numerical_dense_layers: list = field(default_factory = lambda: [
        32
    ])
    layer_numerical_dense_layers: list = field(default_factory = lambda: [
        32
    ])
    
    num_blocks: int = 2
    num_heads: int = 4
    ff_dim: int = 128
    output_dim: int = 64
    dropout_rate: float = 0.1
    
    embedding_outputs: list = field(default_factory = lambda: [
        16,
        16,
        16,
        16
    ])
    
    dense_layers: list = field(default_factory = lambda: [
        128,
        128,
        64
    ])
    
    dense_dropouts: list = field(default_factory = lambda: [
        0.2,
        0.2,
    ])

best_bram_dense_settings = DenseSettings(
    numerical_dense_layers=[16, 32],
    dense_layers=[16, 32, 16, 64, 16, 32, 256, 32],
    dense_dropouts=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)
best_lut_dense_settings = DenseSettings(
    numerical_dense_layers=[16, 16, 64],
    dense_layers=[128, 16, 32, 256, 32],
    dense_dropouts=[0.2, 0.0, 0.0, 0.4]
)
best_overall_dense_settings = DenseSettings(
    numerical_dense_layers=[16, 16, 64, 16],
    dense_layers=[64, 256, 16, 32, 256, 32],
    dense_dropouts=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)

class ModelEmbedding(Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, inputs):
        return inputs

class TransformerBlock(Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        output_dim,
        dropout_rate=0.1
    ):
        super().__init__()
        self.att = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        
        # self.global_avg_pooling = GlobalAveragePooling1D()
        self.out_dense = Dense(output_dim)
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        outputs = self.layernorm2(out1 + ffn_output)
        
        # outputs = self.global_avg_pooling(outputs) # avg of hidden states
        outputs = outputs[:, -1, :] # last hidden state

        outputs = self.out_dense(outputs)
        return outputs

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, position, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angle_rates

def indexed_normalized_mae(index, name='', max_value=200.0):
    def normalized_mae(y_true, y_pred):
        y_true = y_true[..., index]
        y_pred = y_pred[..., index]
        
        # y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        # y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        # print(y_true)
        # print(y_pred)

        diff = tf.abs(y_true - y_pred) / tf.cast(tf.abs(max_value), dtype=tf.float32)

        return 100.0 * tf.reduce_mean(diff, axis=-1)

    if name == '':
        name = index
    normalized_mae.__name__ = f'n_mae_{name}'

    return normalized_mae

def indexed_mape(index, name='', eps=1e-6):
    def mape(y_true, y_pred):
        y_true = y_true[..., index]
        y_pred = y_pred[..., index]
        
        # y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        # y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        # print(y_true)
        # print(y_pred)

        diff = tf.abs(y_true - y_pred) / tf.maximum(tf.abs(y_true), eps)

        return 100.0 * tf.reduce_mean(diff, axis=-1)

    if name == '':
        name = index
    mape.__name__ = f'mape_{name}'

    return mape

def indexed_log_rmse(index, name=''):
    def log_rmse(y_true, y_pred):
        y_true = y_true[..., index]
        y_pred = y_pred[..., index]
        
        squared_log = tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))
        rmsl_error = tf.sqrt(tf.reduce_mean(squared_log, axis=-1))
        
        return rmsl_error

    if name == '':
        name = index
    log_rmse.__name__ = f'log_rmse_{name}'
    
    return log_rmse

def weighted_mape_loss(weights, eps=1e-6):
    def weighted_mape(y_true, y_pred):
        used_weights = [1.0] * y_true.shape[-1]
        for i in range(min(len(weights), len(used_weights))):
            used_weights[i] = weights[i]
        
        mape_error = tf.abs(y_true - y_pred) / tf.maximum(tf.abs(y_true), eps)
        weighted_error = mape_error * used_weights
        
        loss = tf.reduce_mean(weighted_error, axis=-1)
        return loss

    return weighted_mape

def weighted_mae_loss(weights):
    def weighted_mae(y_true, y_pred):
        used_weights = [1.0] * y_true.shape[-1]
        for i in range(min(len(weights), len(used_weights))):
            used_weights[i] = weights[i]

        absolute_error = tf.abs(y_true - y_pred)
        weighted_error = absolute_error * used_weights
        
        loss = tf.reduce_mean(weighted_error, axis=-1)
        return loss

    return weighted_mae

def weighted_huber_loss(weights, delta):
    def weighted_hubber(y_true, y_pred):
        used_weights = [1.0] * y_true.shape[-1]
        for i in range(min(len(weights), len(used_weights))):
            used_weights[i] = weights[i]
        
        absolute_errors = tf.abs(y_true - y_pred)
        quadratic_loss = 0.5 * tf.square(absolute_errors) * used_weights
        linear_loss = delta * (absolute_errors - 0.5 * delta) * used_weights
        
        loss = tf.where(absolute_errors <= delta, quadratic_loss, linear_loss)
        return tf.reduce_mean(loss, axis=-1)

    return weighted_hubber

global_input_shape = ()
seq_input_shape = ()
input_shape = ()
output_shape = ()
def get_model(settings, input_shape=input_shape, global_input_shape=global_input_shape, seq_input_shape=seq_input_shape, output_shape=output_shape, verbose=0):
    # model = Sequential([
    #     Masking(mask_value=0., layer_input_shape=(None, seq_input_shape[-1])),
    #     LSTM(64),
    #     # LSTM(16),
    #     # Dropout(0.2),
    #     Dense(32, activation='relu'),
    #     # Dropout(0.2),
    #     Dense(32, activation='relu'),
    #     Dense(output_shape[-1], activation='relu')
    # ])
    # model = Sequential([
    #     Masking(mask_value=0., layer_input_shape=(None, seq_input_shape[-1])),
    #     LSTM(128, return_sequences=True),
    #     LSTM(16),
    #     # Dropout(0.2),
    #     Dense(64, activation='relu'),
    #     Dense(32, activation='relu'),
    #     # Dropout(0.2),
    #     Dense(32, activation='relu'),
    #     Dense(output_shape[-1], activation='relu')
    # ])
    
    if isinstance(settings, DenseSettings):
        model_inputs = []
    
        strategy_inputs = Input(shape=(1,), name='strategy')
        strategy_embedding = tf.squeeze(Embedding(
            input_dim=len(strategy_map) + 1,
            output_dim=16
        )(strategy_inputs), axis=-2)
        model_inputs.append(strategy_inputs)
        
        precision_inputs = Input(shape=(1,), name='precision')
        precision_embedding = tf.squeeze(Embedding(
            input_dim=len(precision_map) + 1,
            output_dim=16
        )(precision_inputs), axis=-2)
        model_inputs.append(precision_inputs)
        
        board_inputs = Input(shape=(1,), name='board')
        board_embedding = tf.squeeze(Embedding(
            input_dim=len(board_map) + 1,
            output_dim=16
        )(board_inputs), axis=-2)
        model_inputs.append(board_inputs)
        
        numerical_inputs = Input(shape=(input_shape[-1] - len(model_inputs),), name='numerical_in')
        numerical_outputs = numerical_inputs
        for idx, units in enumerate(settings.numerical_dense_layers):
            numerical_outputs = Dense(units, use_bias=False)(numerical_outputs)
        model_inputs.append(numerical_inputs)
        
        x = tf.concat([
            strategy_embedding,
            precision_embedding,
            board_embedding,
            numerical_outputs
        ], axis=-1)
        
        for idx, units in enumerate(settings.dense_layers):
            x = Dense(units, activation=settings.dense_activations[idx])(x)
            if idx < len(settings.dense_dropouts) - 1\
                and settings.dense_dropouts[idx] > 0.0:
                x = Dropout(settings.dense_dropouts[idx])(x)
        
        outputs = Dense(output_shape[-1], activation='relu')(x)

        model = Model(
            inputs=model_inputs,
            outputs=outputs,
            name='NN-estim'
        )

        model.build([None] + list(input_shape))
        if verbose > 0:
            model.summary()
    else:
        global_inputs = []
        seq_inputs = []
        
        # Global inputs (global model/hls info)
        strategy_inputs = Input(shape=(1,), name='strategy_input')
        strategy_embedding = tf.squeeze(Embedding(
            input_dim=len(strategy_map) + 1,
            output_dim=settings.embedding_outputs[1]
        )(strategy_inputs), axis=-2)
        # print(strategy_embedding.shape)
        global_inputs.append(strategy_inputs)
        
        board_inputs = Input(shape=(1,), name='board_input')
        board_embedding = tf.squeeze(Embedding(
            input_dim=len(board_map) + 1,
            output_dim=settings.embedding_outputs[3]
        )(board_inputs), axis=-2)
        # print(board_embedding.shape)
        global_inputs.append(board_inputs)
        
        global_numerical_inputs = Input(shape=(global_input_shape[-1] - len(global_inputs)), name='num_layers_input')
        global_numerical_outputs = global_numerical_inputs
        for idx, units in enumerate(settings.global_numerical_dense_layers):
            global_numerical_outputs = Dense(units)(global_numerical_outputs)
        
        global_inputs.append(global_numerical_inputs)
        
        concat_global_inputs = tf.concat([
            strategy_embedding,
            board_embedding,
            global_numerical_outputs
        ], axis=-1)
        # print(concat_global_inputs.shape)
        
        global_outputs = concat_global_inputs
        for idx, units in enumerate(settings.global_dense_layers):
            global_outputs = Dense(units, activation='relu')(global_outputs)
        
        # Sequential inputs (layer-wise info)
        layer_type_inputs = Input(shape=(None, 1), name='layer_type_input')
        layer_type_embedding = tf.squeeze(Embedding(
            input_dim=len(layer_type_map) + 1,
            output_dim=settings.embedding_outputs[0]
        )(layer_type_inputs), axis=-2)
        # print(layer_type_embedding.shape)
        
        seq_inputs.append(layer_type_inputs)
        
        precision_inputs = Input(shape=(None, 1), name='layer_precision_input')
        precision_embedding = tf.squeeze(Embedding(
            input_dim=len(precision_map) + 1,
            output_dim=settings.embedding_outputs[2]
        )(precision_inputs), axis=-2)
        # print(precision_embedding.shape)
        
        seq_inputs.append(precision_inputs)
        
        layer_numerical_inputs = Input(shape=(None, seq_input_shape[-1] - len(seq_inputs)), name='layer_numerical_input')
        layer_numerical_outputs = layer_numerical_inputs
        for idx, units in enumerate(settings.layer_numerical_dense_layers):
            layer_numerical_outputs = Dense(units)(layer_numerical_outputs)
        # print(layer_numerical_inputs.shape)
        
        seq_inputs.append(layer_numerical_inputs)
        
        concat_seq_inputs = tf.concat([
            layer_type_embedding,
            precision_embedding,
            layer_numerical_outputs
        ], axis=-1)
        # print(concat_seq_inputs.shape)
        masked_inputs = Masking(mask_value=0.)(concat_seq_inputs)
        
        x = masked_inputs
        for idx, units in enumerate(settings.seq_dense_layers):
            x = Dense(units, activation='relu')(x)
        
        assert isinstance(settings, (LSTMSettings, TransformerSettings)),\
            'Argument \'settings\' should be an instance of \'LSTMSettings\' or \'TransformerSettings\''
        
        if isinstance(settings, LSTMSettings):
            for units in settings.lstm_layers[:-1]:
                x = LSTM(units, return_sequences=True)(x)
            x = LSTM(settings.lstm_layers[-1])(x)
        else:
            x = TransformerBlock(
                x.shape[-1],
                settings.num_heads,
                settings.ff_dim,
                settings.output_dim,
                settings.dropout_rate
            )(x)
            # x = Flatten()(x)
        
        x = Concatenate(axis=-1)([x, global_outputs])
        
        for idx, units in enumerate(settings.dense_layers):
            x = Dense(units, activation='relu')(x)
            if idx < len(settings.dense_dropouts) - 1\
                and settings.dense_dropouts[idx] > 0.0:
                x = Dropout(settings.dense_dropouts[idx])(x)
        
        # x1 = Dense(128, activation='relu')(x)
        # x1 = Dense(64, activation='relu')(x1)
        # x1 = Dense(1, activation='relu')(x1)
        
        # x2 = Dense(128, activation='relu')(x)
        # x2 = Dense(64, activation='relu')(x2)
        # x2 = Dense(1, activation='relu')(x2)
        
        # x3 = Dense(128, activation='relu')(x)
        # x3 = Dense(64, activation='relu')(x3)
        # x3 = Dense(1, activation='relu')(x3)
        
        # x4 = Dense(128, activation='relu')(x)
        # x4 = Dense(64, activation='relu')(x4)
        # x4 = Dense(1, activation='relu')(x4)
        
        # x = Concatenate(axis=-1)([x1, x2, x3, x4])
        
        x = Dense(output_shape[-1], activation='relu')(x)
        outputs = x
        
        model = Model(
            inputs=(global_inputs + seq_inputs),
            outputs=outputs,
            name='NN-estim'
        )
        
        model.build([None] + list(global_input_shape) + list(seq_input_shape))
        if verbose > 0:
            model.summary()
            plot_model(model, to_file='transformer_regression.png', show_shapes=True)
    
    return model

def train_dense_model(inputs, targets, input_shape=input_shape, output_shape=output_shape, test=True):
    epochs = 20
    batch_size = 32
    lr = 1e-4
    verbose = 1
    
    train_data, val_data = prepare_simple_dataset(inputs, targets, batch_size)
    
    model = get_model(
        DenseSettings(
            numerical_dense_layers=[64, 16, 64, 64],
            dense_layers=[64, 16, 32, 128, 256, 32, 64],
            dense_activations=['relu', None, None, 'relu', 'tanh', 'tanh', None],
            dense_dropouts=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        input_shape=input_shape,
        output_shape=output_shape,
        verbose=1
    )

    loss_weights = [10, 1, 1, 0.5]
    weighted_mae = weighted_mae_loss(loss_weights)

    metric_names = ['bram', 'dsp', 'ff', 'lut']
    metrics =\
        [indexed_normalized_mae(i, metric_names[i]) for i in range(targets.shape[-1])] +\
        [indexed_mape(i, metric_names[i]) for i in range(targets.shape[-1])]
        # [indexed_log_rmse(i, metric_names[i]) for i in range(targets.shape[-1])]
    
    model.compile(
        optimizer=Adam(lr),
        loss=['mae'],
        metrics=metrics
    )
    
    checkpoint_path = './estimation-checkpoints'
    
    start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('./estimation-logs', start_time)
        
    checkpoint_dir = os.path.join(checkpoint_path, start_time)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_file = os.path.join(checkpoint_dir, 'best-checkpoint.hdf5')
        
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_file,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
        
    def scheduler(epoch, lr):
        if (epoch + 1) % 20 == 0:
            return lr * tf.math.exp(-0.2)
        return lr
    lr_callback = LearningRateScheduler(scheduler)
    
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        write_steps_per_second=False,
        update_freq='epoch',
        embeddings_freq=1,
    )
    
    model.fit(
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_data,
        callbacks=[
            # lr_callback,
            checkpoint_callback,
            tensorboard_callback
        ]
    )

    if test:
        # data_filter.boards = ['zcu102']
        test_inputs, test_targets = simple_data_from_json(
            './datasets/complex/test_dataset.json',
            data_filter
        )
        test_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'strategy': test_inputs[:, 0],
                'precision': test_inputs[:, 1],
                'board': test_inputs[:, 2],
                'numerical_in': test_inputs[:, 3:],
            },
            test_targets
        ))
        test_dataset = test_dataset.shuffle(len(test_dataset)).batch(1)
        
        n_samples = 20
        ground_truth = []
        predictions = []
        # for (inputs, targets) in test_dataset.take(n_samples).as_numpy_iterator():
        for (inputs, targets) in test_dataset.as_numpy_iterator():
            # print(inputs)
            print(targets)
            ground_truth.append(targets)
            
            predictions.append(model.predict(inputs, 0))
            print(predictions[-1])
        
        ground_truth = np.squeeze(np.asarray(ground_truth))
        predictions = np.squeeze(np.asarray(predictions))
        
        prediction_errors = np.abs(predictions - ground_truth)
        print(prediction_errors)
        
        bin_width = 5
        num_bins = int(np.ceil(200. / bin_width))

        plt.figure(figsize=(8, 6))
        plt.hist(prediction_errors, bins=num_bins, range=(0, 200.), edgecolor='black')
        plt.xlabel('Error (%)')
        plt.ylabel('Frequency')
        plt.title('LUT MAE (%)')
        plt.xticks(np.arange(0, 200. + 1, step=bin_width))
        
        plt.show()

    return model

batch_size = 32
def dense_model_builder_gen(input_shape, output_shape):
    def dense_model_builder(hp):
        batch_size = hp.Choice(name='batch_size', values=[16, 32, 64, 128])
        settings = DenseSettings()
        
        numerical_dense_depth = hp.Int(name='numerical_dense_depth', min_value=1, max_value=4, step=1)
        settings.numerical_dense_layers = [
            hp.Choice(name=f'numerical_dense_{i}', values=[16, 32, 64])\
                for i in range(numerical_dense_depth)
        ]
        
        dense_depth = hp.Int(name='dense_depth', min_value=2, max_value=8, step=1)
        settings.dense_layers = [
            hp.Choice(name=f'dense_{i}', values=[16, 32, 64, 128, 256])\
                for i in range(dense_depth)
        ]
        settings.dense_activations = [
            hp.Choice(name=f'dense_act_{i}', values=['None', 'relu', 'tanh'])
                for i in range(dense_depth)
        ]
        for i in range(len(settings.dense_activations)):
            if settings.dense_activations[i] == 'None':
                settings.dense_activations[i] = None
        
        dropout_count = hp.Int(name='dropout_count', min_value=0, max_value=dense_depth)
        settings.dense_dropouts = [
            hp.Float(name=f'dense_dropout_{i}', min_value=0.0, max_value=0.4, step=0.1)\
                for i in range(dropout_count)
        ]
        
        model = get_model(settings, input_shape=input_shape, output_shape=output_shape, verbose=0)
        
        # lr = hp.Float(name='learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        lr = hp.Choice(name='learning_rate', values=[1e-4, 1e-3])
        optimizer = hp.Choice(name='optimizer', values=['SGD', 'Adam'])
        if optimizer == 'Adam':
            optimizer = Adam(lr)
        elif optimizer == 'SGD':
            optimizer = SGD(lr)
        
        metric_names = ['bram', 'dsp', 'ff', 'lut']
        metrics =\
            [indexed_normalized_mae(i, metric_names[i]) for i in range(1)] +\
            [indexed_mape(i, metric_names[i]) for i in range(1)] +\
            [indexed_log_rmse(i, metric_names[i]) for i in range(1)]
        
        model.compile(
            optimizer=optimizer,
            loss=['mae'],
            metrics=metrics
        )
        
        return model

    return dense_model_builder

def model_builder(hp):
    settings = None

    batch_size = hp.Choice(name='batch_size', values=[16, 32, 64, 128])

    # model_head = hp.Choice(name='head', values=['lstm', 'transformer'])
    model_head = hp.Choice(name='model_head', values=['transformer'])
    if model_head == 'lstm':
        settings = LSTMSettings()
        
        lstm_depth = hp.Int(name='lstm_depth', min_value=1, max_value=8)
        settings.lstm_layers = [
            hp.Int(name=f'lstm_units_{i}', min_value=32, max_value=128, step=32)\
                for i in range(lstm_depth)
        ]
    else:
        settings = TransformerSettings(
            num_heads=hp.Int(name='num_heads', min_value=4, max_value=12, step=2),
            ff_dim=hp.Int(name='ff_dim', min_value=32, max_value=256, step=32),
            output_dim=hp.Int(name='output_dim', min_value=32, max_value=256, step=32),
            dropout_rate=hp.Float(name='transformer_dropout', min_value=0.0, max_value=0.6, step=0.2)
        )
    
    embedding_count = 4
    settings.embedding_outputs = [
        hp.Int(name=f'embedding_output_{i}', min_value=8, max_value=32, step=8)\
            for i in range(embedding_count)
    ]
    
    global_depth = hp.Int(name='global_depth', min_value=1, max_value=4, step=1)
    settings.global_dense_layers = [
        hp.Int(name=f'global_units_{i}', min_value=32, max_value=256, step=32)\
            for i in range(global_depth)
    ]
    
    seq_depth = hp.Int(name='seq_depth', min_value=1, max_value=4, step=1)
    settings.seq_dense_layers = [
        hp.Int(name=f'seq_units_{i}', min_value=32, max_value=256, step=32)\
            for i in range(seq_depth)
    ]
    
    global_numerical_depth = hp.Int(name='global_numerical_depth', min_value=1, max_value=4, step=1)
    settings.global_numerical_dense_layers = [
        hp.Int(name=f'global_numerical_units_{i}', min_value=8, max_value=32, step=8)\
            for i in range(global_numerical_depth)
    ]
    
    layer_numerical_depth = hp.Int(name='layer_numerical_depth', min_value=1, max_value=4, step=1)
    settings.layer_numerical_dense_layers = [
        hp.Int(name=f'layer_numerical_units_{i}', min_value=8, max_value=32, step=8)\
            for i in range(layer_numerical_depth)
    ]
    
    dense_depth = hp.Int(name='dense_depth', min_value=4, max_value=12, step=2)
    settings.dense_layers = [
        hp.Int(name=f'dense_units_{i}', min_value=32, max_value=256, step=32)\
            for i in range(dense_depth)
    ]
    
    dropout_count = hp.Int(name='dropout_count', min_value=0, max_value=dense_depth)
    settings.dense_dropouts = [
        hp.Float(name=f'dense_dropout_{i}', min_value=0.0, max_value=0.4, step=0.1)\
            for i in range(dropout_count)
    ]
    
    model = get_model(settings, verbose=0)
    
    # lr = hp.Float(name='learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    lr = hp.Choice(name='learning_rate', values=[1e-4, 1e-3])
    optimizer = hp.Choice(name='optimizer', values=['Adam'])
    if optimizer == 'Adam':
        optimizer = Adam(lr)
    elif optimizer == 'SGD':
        optimizer = SGD(lr)
    
    # loss = hp.Choice(name='loss', values=['huber', 'mae'])
    # if loss == 'huber':
    #     loss = huber
    loss = weighted_mae_loss([10, 15, 10, 1])
    metric_names = ['bram', 'dsp', 'ff', 'lut']
    metrics =\
        [indexed_normalized_mae(i, metric_names[i]) for i in range(1)] +\
        [indexed_mape(i, metric_names[i]) for i in range(1)] +\
        [indexed_log_rmse(i, metric_names[i]) for i in range(1)]
    
    model.compile(
        optimizer=optimizer,
        loss=['mae'],
        metrics=metrics
    )
    
    return model

def prepare_simple_dataset(inputs, targets, batch_size, val_ratio=0.2, train_repeats=10):
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'strategy': inputs[:, 0],
            'precision': inputs[:, 1],
            'board': inputs[:, 2],
            'numerical_in': inputs[:, 3:],
        },
        targets
    ))

    train_data = val_data = None
    if val_ratio > 0.0:    
        train_data = dataset.shuffle(len(dataset))
        val_data = dataset.take(int(len(dataset) * val_ratio))
        val_data = val_data.batch(batch_size)
    train_data = dataset.skip(int(len(dataset) * val_ratio)).repeat(train_repeats)\
        .shuffle(int((1 - val_ratio) * len(dataset))).batch(batch_size)
    # train_data = dataset.skip(int(len(dataset) * val_ratio))\
    #     .shuffle(int((1 - val_ratio) * len(dataset))).batch(batch_size)

    return train_data, val_data

def prepare_seq_dataset(global_inputs, seq_inputs, targets, batch_size, repeats=10):
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'strategy_input': global_inputs[:, 0],
            'board_input': global_inputs[:, 1],
            'num_layers_input': global_inputs[:, 2:],
            'layer_precision_input': seq_inputs[:, :, 0],
            'layer_type_input': seq_inputs[:, :, 1],
            'layer_numerical_input': seq_inputs[:, :, 2:]
        },
        targets
    ))
    
    # train_data = dataset.shuffle(len(dataset)).repeat(10).batch(batch_size)
    val_ratio = 0.2
    train_data = dataset.shuffle(len(dataset))
    val_data = dataset.take(int(len(dataset) * val_ratio))
    val_data = val_data.batch(batch_size)
    train_data = dataset.skip(int(len(dataset) * val_ratio)).repeat(repeats)\
        .shuffle(int((1 - val_ratio) * len(dataset))).batch(batch_size)
    
    return train_data, val_data

def dense_hp_search(inputs, targets, input_shape=input_shape, output_shape=output_shape):
    hp = kt.HyperParameters()
        
    start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('./hp-search/logs', start_time)
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        write_steps_per_second=False,
        update_freq='epoch',
        embeddings_freq=1,
    )
    
    dense_model_builder = dense_model_builder_gen(input_shape, output_shape)
    
    # tuner = kt.Hyperband(
    #     hypermodel=dense_model_builder,
    #     objective='val_loss',
    #     max_epochs=20,
    #     factor=3,
    #     hyperband_iterations=1,
    #     directory='./hp-search',
    #     project_name=start_time,
    # )
    tuner = kt.BayesianOptimization(
        hypermodel=dense_model_builder,
        objective='val_loss',
        max_trials=100,
        directory='./hp-search',
        project_name=start_time,
    )
    tuner.search_space_summary()
    
    train_data, val_data = prepare_simple_dataset(inputs, targets, batch_size)
    tuner.search(
        train_data,
        validation_data=val_data,
        callbacks=[tensorboard_callback],
        batch_size=batch_size
    )
    
    return tuner

def hp_search(global_inputs, seq_inputs, targets):
    hp = kt.HyperParameters()
        
    start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('./hp-search/logs', start_time)
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        write_steps_per_second=False,
        update_freq='epoch',
        embeddings_freq=1,
    )
    
    tuner = kt.Hyperband(
        hypermodel=model_builder,
        objective='val_loss',
        max_epochs=20,
        factor=3,
        hyperband_iterations=1,
        directory='./hp-search',
        project_name=start_time,
    )
    # tuner = kt.BayesianOptimization(
    #     hypermodel=model_builder,
    #     objective='val_loss',
    #     max_trials=100,
    #     directory='./hp-search',
    #     project_name=start_time,
    # )
    tuner.search_space_summary()
    
    train_data, val_data = prepare_seq_dataset(global_inputs, seq_inputs, targets, batch_size)
    tuner.search(
        train_data,
        validation_data=val_data,
        callbacks=[tensorboard_callback],
        batch_size=batch_size
    )
    
    return tuner
    
def train(global_inputs, seq_inputs, targets, global_input_shape=global_input_shape, seq_input_shape=seq_input_shape, output_shape=output_shape):
    epochs = 50
    batch_size = 16
    lr = 1e-4
    verbose = 1
    train = True
    
    train_data, val_data = prepare_seq_dataset(global_inputs, seq_inputs, targets, batch_size, repeats=1)
        
    loss_weights = [10, 100, 10, 0.1]
    weighted_mae = weighted_mae_loss(loss_weights)
    weighted_huber = weighted_huber_loss([], 0.1)
    
    metric_names = ['bram', 'dsp', 'ff', 'lut']
    metrics =\
        [indexed_normalized_mae(i, metric_names[i]) for i in range(targets.shape[-1])] +\
        [indexed_mape(i, metric_names[i]) for i in range(targets.shape[-1])] +\
        [indexed_log_rmse(i, metric_names[i]) for i in range(targets.shape[-1])]
    
    model = get_model(
        # settings=LSTMSettings(
        #     lstm_layers=[96, 64, 96, 128, 96, 32, 32, 32],
        #     embedding_outputs=[16, 12, 16, 12],
        #     dense_layers= [96, 16, 16, 80, 128, 128],
        #     dense_dropouts= [0.2, 0.3]
        # ),
        settings=TransformerSettings(
            global_dense_layers=[128, 192, 192],
            seq_dense_layers=[32, 64, 96],
            global_numerical_dense_layers=[16, 8],
            layer_numerical_dense_layers=[32],
            num_heads=8,
            ff_dim=256,
            output_dim=192,
            dropout_rate=0.2,
            embedding_outputs=[24, 24, 16, 8],
            dense_layers=[192, 128, 64, 32, 64, 128, 256, 32],
            dense_dropouts=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        global_input_shape=global_input_shape,
        seq_input_shape=seq_input_shape,
        output_shape=output_shape,
        verbose=verbose
    )
    model.compile(
        optimizer=Adam(lr),
        loss=['mae'],
        metrics=metrics
    )
    
    checkpoint_path = './estimation-checkpoints'
    
    if train:
        start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join('./estimation-logs', start_time)
        
        checkpoint_dir = os.path.join(checkpoint_path, start_time)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, 'best-checkpoint.hdf5')
        
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_file,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
        
        def scheduler(epoch, lr):
            if (epoch + 1) % 20 == 0:
                return lr * tf.math.exp(-0.2)
            return lr
        lr_callback = LearningRateScheduler(scheduler)
        
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            write_steps_per_second=False,
            update_freq='epoch',
            embeddings_freq=1,
        )
        
        model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=[
                # lr_callback,
                checkpoint_callback,
                tensorboard_callback
            ]
        )
    else:
        checkpoint_file = os.path.join(
            checkpoint_path,
            '20240325-221401',
            'best-checkpoint.hdf5'
        )

    model.load_weights(checkpoint_file)

    global_inputs, seq_inputs, targets = padded_data_from_json(
        './datasets/complex/test_dataset.json',
        data_filter
    )
    test_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'strategy_input': global_inputs[:, 0],
            'board_input': global_inputs[:, 1],
            'num_layers_input': global_inputs[:, 2:],
            'layer_precision_input': seq_inputs[:, :, 0],
            'layer_type_input': seq_inputs[:, :, 1],
            'layer_numerical_input': seq_inputs[:, :, 2:]
        },
        targets
    ))
    test_dataset = test_dataset.shuffle(len(test_dataset)).batch(1)
    
    n_samples = 20
    ground_truth = []
    predictions = []
    # for (inputs, targets) in test_dataset.take(n_samples).as_numpy_iterator():
    for (inputs, targets) in test_dataset.as_numpy_iterator():
        # print(inputs)
        print(targets)
        ground_truth.append(targets)
        
        predictions.append(model.predict(inputs, 0))
        print(predictions[-1])
    
    ground_truth = np.squeeze(np.asarray(ground_truth))
    predictions = np.squeeze(np.asarray(predictions))
    
    prediction_errors = np.abs(predictions - ground_truth)
    print(prediction_errors)
    
    bin_width = 5
    num_bins = int(np.ceil((prediction_errors.max()) / bin_width))

    plt.figure(figsize=(8, 6))
    plt.hist(prediction_errors, bins=num_bins, range=(0, prediction_errors.max()), edgecolor='black')
    plt.xlabel('Error (%)')
    plt.ylabel('Frequency')
    plt.title('LUT MAE (%)')
    plt.xticks(np.arange(0, np.floor(prediction_errors.max()) + 1, step=bin_width))
    
    plt.show()
    
    # n_features = ground_truth.shape[-1]
    # feature_names = [
    #     'BRAM',
    #     'DSP',
    #     'FF',
    #     'LUT',
    # ]

    # sample_indices = np.arange(n_samples)

    # # Plot grouped bar chart
    # bar_width = 0.35
    # opacity = 0.5

    # fig, ax = plt.subplots(figsize=(10, 6))

    # for i in range(2):
    #     bar_name = 'Real' if i == 0 else 'Predicted'
    #     values = ground_truth if i == 0 else predictions
        
    #     ax.bar(sample_indices - 3*bar_width/4, values[:, 0], bar_width/2, alpha=opacity, label=f'{bar_name} {feature_names[0]}')
    #     ax.bar(sample_indices - bar_width/4, values[:, 1], bar_width/2, alpha=opacity, label=f'{bar_name} {feature_names[1]}')
    #     ax.bar(sample_indices + bar_width/4, values[:, 2], bar_width/2, alpha=opacity, label=f'{bar_name} {feature_names[2]}')
    #     ax.bar(sample_indices + 3*bar_width/4, values[:, 3], bar_width/2, alpha=opacity, label=f'{bar_name} {feature_names[3]}')
    
    # ax.set_xlabel('Sample Index')
    # ax.set_ylabel('Values')
    # ax.set_title('Real vs. Predicted Values for Each Feature')
    # ax.set_xticks(sample_indices)
    # ax.legend()

    # plt.tight_layout()
    # plt.show()

    return model

def model_predict(predictor, model_to_predict, hls_config, board):
    keras_config = parse_keras_config(model_to_predict, hls_config['reuse_factor'])
    hls_precision = hls_config['Precision']
    hls_strategy = hls_config['Strategy']
    
    layers_data = np.asarray([
        [precision_map[hls_precision.lower()]] + parse_layer_data(layer_config)\
        for layer_config in keras_config
    ])
    # layers_data = layers_data[np.newaxis, ...]
    print(layers_data.shape)
    print(layers_data)
    
    inputs = {
        'strategy_input': strategy_map[hls_strategy.lower()],
        'board_input': board_map[board.lower()],
        'num_layers_input': len(keras_config),
        'layer_precision_input': layers_data[..., 0],
        'layer_type_input': layers_data[..., 1],
        'layer_numerical_input': layers_data[..., 2:]
    }    
    inputs = {
        key: tf.convert_to_tensor([value]) for key, value in inputs.items()
    }
    
    return predictor.predict(inputs, 0)

if __name__ == '__main__':
    data_filter = NetworkDataFilter(
        exclude_layers=['Concatenate', 'Add'],
        max_output_size=200,
        # max_output_size=100,
        # max_layers=8,
        # strategies = ['Resource'],
        # boards=['pynq-z2']
    )
    
    use_sequence_data = True

    if use_sequence_data:
        # global_inputs, seq_inputs, targets = padded_data_from_json(
        #     './datasets/complex/dataset-*.json',
        #     data_filter
        # )
        global_inputs, seq_inputs, targets = padded_data_from_json(
            './datasets/complex/augmented_train_5p.json',
            data_filter
        )
        print(global_inputs.shape)
        print(seq_inputs.shape)
        print(targets.shape)
        
        global_input_shape = global_inputs.shape[1:]
        seq_input_shape = seq_inputs.shape[1:]
        output_shape = targets.shape[1:]
        
        model = train(
            global_inputs,
            seq_inputs,
            targets,
            global_input_shape,
            seq_input_shape,
            output_shape
        )
        
        # tuner = hp_search(global_inputs, seq_inputs, targets)
        # print(tuner.results_summary())
    else:
        inputs, targets = simple_data_from_json(
            './datasets/complex/dataset-*.json',
            data_filter
        )
        
        input_shape = inputs.shape[1:]
        output_shape = targets.shape[1:]
        
        model = train_dense_model(inputs, targets)
        
        # tuner = dense_hp_search(inputs, targets)
        # print(tuner.results_summary())

    input_size = 16
    inputs = Input(input_size,)
    x = Dense(32, use_bias=True)(inputs)
    x = Activation('tanh')(x)
    x = Dense(64, use_bias=True)(x)
    x = Activation('relu')(x)
    x = Dense(32, use_bias=True)(x)
    x = Activation('relu')(x)
    x = Dense(3, use_bias=True)(x)
    outputs = Activation('relu')(x)
    
    model_to_predict = Model(inputs=inputs, outputs=outputs)
    model_to_predict.build([None, input_size])

    hls_config = {
        'Precision': 'ap_fixed<8, 3>',
        'Strategy': 'Resource',
        'reuse_factor': 16,
    }
    target_board = 'pynq-z2'
    
    hls_model, _ = to_hls(
        model_to_predict,
        './hls4ml_prj',
        {'Model': hls_config['Precision']},
        strategy=hls_config['Strategy'],
        reuse_factor=hls_config['reuse_factor'],
        board=target_board,
    )
    result = hls_model.build(
        csim=False,
        synth=True,
        export=False,
        bitfile=False
    )
    print(f'Truth: {result}')
    
    prediction = model_predict(model, model_to_predict, hls_config, target_board)
    print(f'Prediction: {prediction}')
