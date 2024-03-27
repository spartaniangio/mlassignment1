import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)
num_samples = 200
num_features = 4
feature_names = ['Protocol_Type', 'Source_IP', 'Destination_IP', 'Packet_Length']

data = np.random.rand(num_samples, num_features)

data[:, 1] = data[:, 0] + np.random.normal(0, 0.05, num_samples)
data[:, 2] = data[:, 0] + np.random.normal(0, 0.05, num_samples)

df = pd.DataFrame(data, columns=feature_names)

df.to_csv('network_data.csv', index=False)
correlation_matrix = np.corrcoef(data, rowvar=False)

correlation_df = pd.DataFrame(correlation_matrix, columns=feature_names, index=feature_names)
correlation_df.to_csv('correlation_data.csv', index=True)

sns.set(style="white")
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, xticklabels=feature_names, yticklabels=feature_names)
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.pdf')
plt.close()

print("The End")