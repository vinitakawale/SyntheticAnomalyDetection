import matplotlib.pyplot as plt

def plot_anomaly(data_converted, anomalous):
    # Plot the data with anomalies marked in red
  plt.figure(figsize=(16, 8))
  plt.plot(data_converted['timestamp'],
         data_converted['value'])
  plt.plot(data_converted['timestamp'][anomalous],
         data_converted['value'][anomalous], 'ro')
  plt.title('Anomaly Detection')
  plt.xlabel('Time')
  plt.ylabel('Value')
  plt.show()