# Exclude datetime column again
import tensorflow as tf
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
 
def Autoencoder(data_converted):
  data_tensor = tf.convert_to_tensor(data_converted.drop(
    'timestamp', axis=1).values, dtype=tf.float32)


  # Define the autoencoder model
  input_dim = data_converted.shape[1] - 1
  encoding_dim = 10

  input_layer = Input(shape=(input_dim,))
  encoder = Dense(encoding_dim, activation='relu')(input_layer)
  decoder = Dense(input_dim, activation='relu')(encoder)
  autoencoder = Model(inputs=input_layer, outputs=decoder)

  # Compile and fit the model
  autoencoder.compile(optimizer='adam', loss='mse')
  autoencoder.fit(data_tensor, data_tensor, epochs=50,
                batch_size=32, shuffle=True)

  # Calculate the reconstruction error for each data point
  reconstructions = autoencoder.predict(data_tensor)
  mse = tf.reduce_mean(tf.square(data_tensor - reconstructions),
                     axis=1)
  anomaly_scores = pd.Series(mse.numpy(), name='anomaly_scores')
  anomaly_scores.index = data_converted.index

  return anomaly_scores

