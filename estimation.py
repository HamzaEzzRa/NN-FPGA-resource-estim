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
from tensorflow.keras.optimizers import SGD, Adam

from network_parser import *


@dataclass
class LSTMSettings:
    lstm_layers: list = field(default_factory = lambda: [
        128,
        32
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
    num_heads: int = 4
    ff_dim: int = 128
    output_dim: int = 64
    dropout_rate: float = 0.1
    
    dense_layers: list = field(default_factory = lambda: [
        128,
        128,
        64
    ])
    
    dense_dropouts: list = field(default_factory = lambda: [
        0.2,
        0.2,
    ])

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
        
        self.global_avg_pooling = GlobalAveragePooling1D()
        self.out_dense = Dense(output_dim)
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        outputs = self.layernorm2(out1 + ffn_output)
        outputs = self.global_avg_pooling(outputs)

        outputs = self.out_dense(outputs)
        return outputs

def mape(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    
    # print(y_true)
    # print(y_pred)

    epsilon = 1
    diff = tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), epsilon))

    return 100.0 * tf.reduce_mean(diff)

input_shape = ()
output_shape = ()
def get_model(settings, verbose=0):
    # model = Sequential([
    #     Masking(mask_value=0., input_shape=(None, input_shape[-1])),
    #     LSTM(64),
    #     # LSTM(16),
    #     # Dropout(0.2),
    #     Dense(32, activation='relu'),
    #     # Dropout(0.2),
    #     Dense(32, activation='relu'),
    #     Dense(output_shape[-1], activation='relu')
    # ])
    # model = Sequential([
    #     Masking(mask_value=0., input_shape=(None, input_shape[-1])),
    #     LSTM(128, return_sequences=True),
    #     LSTM(16),
    #     # Dropout(0.2),
    #     Dense(64, activation='relu'),
    #     Dense(32, activation='relu'),
    #     # Dropout(0.2),
    #     Dense(32, activation='relu'),
    #     Dense(output_shape[-1], activation='relu')
    # ])
    
    inputs = []
    
    layer_type_inputs = Input(shape=(None, 1))
    layer_type_embedding = tf.squeeze(Embedding(
        input_dim=len(layer_type_map) + 1,
        output_dim=16
    )(layer_type_inputs), axis=-2)
    # print(layer_type_embedding.shape)
    
    inputs.append(layer_type_inputs)

    strategy_inputs = Input(shape=(None, 1))
    strategy_embedding = tf.squeeze(Embedding(
        input_dim=len(strategy_map) + 1,
        output_dim=16
    )(strategy_inputs), axis=-2)
    # print(strategy_embedding.shape)
    
    inputs.append(strategy_inputs)
    
    precision_inputs = Input(shape=(None, 1))
    precision_embedding = tf.squeeze(Embedding(
        input_dim=len(precision_map) + 1,
        output_dim=16
    )(precision_inputs), axis=-2)
    # print(precision_embedding.shape)
    
    inputs.append(precision_inputs)
    
    board_inputs = Input(shape=(None, 1))
    board_embedding = tf.squeeze(Embedding(
        input_dim=len(board_map) + 1,
        output_dim=16
    )(board_inputs), axis=-2)
    # print(board_embedding.shape)
    
    inputs.append(board_inputs)
    
    layer_numerical_inputs = Input(shape=(None, input_shape[-1] - len(inputs)))
    # print(layer_numerical_inputs.shape)
    
    inputs.append(layer_numerical_inputs)
    
    concat_inputs = tf.concat([
        layer_type_embedding,
        strategy_embedding,
        precision_embedding,
        board_embedding,
        layer_numerical_inputs
    ], axis=-1)
    # print(concat_inputs.shape)
    masked_inputs = Masking(mask_value=0.)(concat_inputs)
    
    x = masked_inputs
    
    assert isinstance(settings, (LSTMSettings, TransformerSettings)),\
        'Argument \'settings\' should be an instance of \'LSTMSettings\' or \'TransformerSettings\''
    
    if isinstance(settings, LSTMSettings):
        for units in settings.lstm_layers[:-1]:
            x = LSTM(units, return_sequences=True)(x)
        x = LSTM(settings.lstm_layers[-1])(x)
    else:
        x = TransformerBlock(
            masked_inputs.shape[-1],
            settings.num_heads,
            settings.ff_dim,
            settings.output_dim,
            settings.dropout_rate
        )(x)
        x = Flatten()(x)
    
    for idx, units in enumerate(settings.dense_layers):
        x = Dense(units, activation='relu')(x)
        if idx < len(settings.dense_dropouts) - 1:
            x = Dropout(settings.dense_dropouts[idx])(x)
    
    x = Dense(output_shape[-1], activation='relu')(x)
    outputs = x
    
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name='NN-estim'
    )
    
    model.build([None] + list(input_shape))
    if verbose > 0:
        model.summary()
    
    return model

def model_builder(hp):
    settings = None

    model_head = hp.Choice(name='head', values=['lstm', 'transformer'])
    if model_head == 'lstm':
        settings = LSTMSettings()
        
        lstm_depth = hp.Int(name='lstm_depth', min_value=1, max_value=6)
        settings.lstm_layers = [
            hp.Int(name=f'lstm_units_{i}', min_value=16, max_value=128, step=16)\
                for i in range(lstm_depth)
        ]
    else:
        settings = TransformerSettings(
            num_heads=hp.Int(name='num_heads', min_value=1, max_value=4),
            ff_dim=hp.Int(name='ff_dim', min_value=16, max_value=128, step=16),
            output_dim=hp.Int(name='output_dim', min_value=16, max_value=128, step=16),
            dropout_rate=hp.Float(name='transformer_dropout', min_value=0.0, max_value=0.5, step=0.1)
        )
    
    dense_depth = hp.Int(name='dense_depth', min_value=1, max_value=6)
    settings.dense_layers = [
        hp.Int(name=f'dense_units_{i}', min_value=16, max_value=128, step=16)\
            for i in range(dense_depth)
    ]
    
    dropout_count = hp.Int(name='dropout_count', min_value=0, max_value=dense_depth)
    settings.dense_dropouts = [
        hp.Float(name=f'dense_dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)\
            for i in range(dropout_count)
    ]
    
    model = get_model(settings, verbose=0)
    
    lr = hp.Float(name='learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    optimizer = hp.Choice(name='optimizer', values=['Adam', 'SGD'])
    if optimizer == 'Adam':
        optimizer = Adam(lr)
    elif optimizer == 'SGD':
        optimizer = SGD(lr)
    
    model.compile(
        optimizer=optimizer,
        loss=hp.Choice(name='loss', values=['mse', 'mae']),
        metrics=[mape]
    )
    
    return model

def prepare_dataset(inputs, targets, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_1": inputs[:, :, 0],
            "input_2": inputs[:, :, 1],
            "input_3": inputs[:, :, 2],
            "input_4": inputs[:, :, 3],
            "input_5": inputs[:, :, 4:]
        },
        targets
    ))
    train_data = dataset.shuffle(len(dataset)).repeat(10).batch(batch_size)

    test_inputs, test_targets = padded_data_from_json('./datasets/mehdi-dataset.json')
    test_dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_1": test_inputs[:, :, 0],
            "input_2": test_inputs[:, :, 1],
            "input_3": test_inputs[:, :, 2],
            "input_4": test_inputs[:, :, 3],
            "input_5": test_inputs[:, :, 4:]
        },
        test_targets
    ))
    val_data = test_dataset.shuffle(len(test_dataset)).batch(batch_size//2)
    
    return train_data, val_data

def hp_search(inputs, targets):
    hp = kt.HyperParameters()
    
    batch_size = hp.Int(name='batch_size', min_value=16, max_value=128, step=16)
    train_data, val_data = prepare_dataset(inputs, targets, batch_size)
    
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
        objective='val_mape',
        max_epochs=50,
        factor=3,
        hyperband_iterations=2,
        directory='./hp-search',
        project_name=start_time,
    )
    tuner.search_space_summary()
    
    tuner.search(
        train_data,
        validation_data=val_data,
        callbacks=[tensorboard_callback]
    )
    
    return tuner
    
def train(inputs, targets):
    epochs = 100
    batch_size = 32
    lr = 1e-3
    verbose = 1
    train = True
    
    train_data, val_data = prepare_dataset(inputs, targets, batch_size)
    
    checkpoint_path = './estimation-checkpoints'
    
    model = get_model(
        settings=LSTMSettings(),
        verbose=verbose
    )
    model.compile(
        optimizer=Adam(lr),
        loss='mae',
        metrics=[mape]
    )
    
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
            if (epoch + 1) % 30 == 0:
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
                lr_callback,
                checkpoint_callback,
                tensorboard_callback
            ]
        )
    else:
        checkpoint_file = os.path.join(
            checkpoint_path,
            '20231228-180403',
            'best-checkpoint.hdf5'
        )

    model.load_weights(checkpoint_file)

    inputs, targets = padded_data_from_json('./datasets/mehdi-dataset.json')
    test_dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_1": inputs[:, :, 0],
            "input_2": inputs[:, :, 1],
            "input_3": inputs[:, :, 2],
            "input_4": inputs[:, :, 3],
            "input_5": inputs[:, :, 4:]
        },
        targets
    ))
    test_dataset = test_dataset.shuffle(len(test_dataset)).batch(1)
    
    n_samples = 20
    ground_truth = []
    predictions = []
    for (inputs, targets) in test_dataset.take(n_samples).as_numpy_iterator():
        # print(inputs)
        print(targets)
        ground_truth.append(targets)
        
        predictions.append(model.predict(inputs, 0))
        print(predictions[-1])
    
    ground_truth = np.squeeze(np.asarray(ground_truth))
    predictions = np.squeeze(np.asarray(predictions))
    
    n_features = ground_truth.shape[-1]
    feature_names = [
        'BRAM',
        'DSP',
        'FF',
        'LUT',
    ]

    sample_indices = np.arange(n_samples)

    # Plot grouped bar chart
    bar_width = 0.35
    opacity = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(2):
        bar_name = 'Real' if i == 0 else 'Predicted'
        values = ground_truth if i == 0 else predictions
        
        ax.bar(sample_indices - 3*bar_width/4, values[:, 0], bar_width/2, alpha=opacity, label=f'{bar_name} {feature_names[0]}')
        ax.bar(sample_indices - bar_width/4, values[:, 1], bar_width/2, alpha=opacity, label=f'{bar_name} {feature_names[1]}')
        ax.bar(sample_indices + bar_width/4, values[:, 2], bar_width/2, alpha=opacity, label=f'{bar_name} {feature_names[2]}')
        ax.bar(sample_indices + 3*bar_width/4, values[:, 3], bar_width/2, alpha=opacity, label=f'{bar_name} {feature_names[3]}')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Values')
    ax.set_title('Real vs. Predicted Values for Each Feature')
    ax.set_xticks(sample_indices)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    inputs, targets = padded_data_from_json('./datasets/dataset-*.json')
    # print(inputs.shape)
    # print(targets.shape)
    
    input_shape = inputs.shape[1:]
    output_shape = targets.shape[1:]
    
    # train(inputs, targets)
    tuner = hp_search(inputs, targets)
    print(tuner.results_summary())
