import os
from datetime import datetime

import tensorflow as tf
from network_parser import layer_type_map, load_from_json
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    Layer,
    LayerNormalization,
    Masking,
    MultiHeadAttention,
)
from tensorflow.keras.optimizers import SGD, Adam


class ModelEmbedding(Layer):
    def __init__(self, maxlen, layer_type_size, layer_type_dim):
        super().__init__()
        
        self.layer_emb = Embedding(
            input_dim
        )
        self.token_emb = Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
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
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def get_model(input_shape, verbose=0):
    # model = Sequential([
    #     Masking(mask_value=0., input_shape=(None, input_shape[-1])),
    #     LSTM(64),
    #     # LSTM(16),
    #     # Dropout(0.2),
    #     Dense(32, activation='relu'),
    #     # Dropout(0.2),
    #     Dense(32, activation='relu'),
    #     Dense(4, activation='relu')
    # ])
    model = Sequential([
        Masking(mask_value=0., input_shape=(None, input_shape[-1])),
        LSTM(128, return_sequences=True),
        LSTM(16),
        # Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        # Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(4, activation='relu')
    ])
    
    model.build([None] + list(input_shape))
    if verbose > 0:
        model.summary()
    
    return model

if __name__ == '__main__':
    inputs, targets = load_from_json('./dataset.json')
    print(inputs.shape)
    print(targets.shape)
    
    epochs = 100
    batch_size = 16
    lr = 1e-3
    verbose = 1
    train = True
    
    checkpoint_path = './estimation-checkpoints'
    
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.batch(batch_size).shuffle(200)

    # for batch in dataset.take(1).as_numpy_iterator():
    #     for inputs, target in zip(batch[0], batch[1]):
    #         print(inputs)
    #         print(target)
            
    #         break

    val_data = dataset.take(int(len(dataset) * 0.15))
    train_data = dataset.skip(int(len(dataset) * 0.15))
    
    model = get_model(inputs.shape[1:], verbose)
    model.compile(
        optimizer=Adam(lr),
        loss='mae',
        metrics=['mape']
    )
    
    if train:
        # model.compile(
        #     optimizer=Adam(lr),
        #     loss='mae',
        #     metrics=['mape']
        # )
        
        start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
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
            if (epoch + 1) % 100 == 0:
                return lr * tf.math.exp(-0.5)
            return lr
        lr_callback = LearningRateScheduler(scheduler)
        
        model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=[lr_callback, checkpoint_callback]
        )
    else:
        checkpoint_file = os.path.join(
            checkpoint_path,
            '20231215-103645',
            'best-checkpoint.hdf5'
        )

    model.load_weights(checkpoint_file)

    inputs, targets = load_from_json('./dataset_100.json')
    test_dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    # test_dataset = test_dataset.batch(batch_size)
    
    # loss = model.evaluate(test_dataset, batch_size=8)
