import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

path = "/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/han/dataframe (not uploaded)"
df = pd.read_csv(path)

# Filter the DataFrame as the engine is running
filtered_df = df[(df['fr_eng'] > 0) & (df['fr_eng_ecs'] > 0)]

# Choose a variable as y ('pr_baro') and other variables as X
df_sampled = filtered_df.dropna()
y = df_sampled['pr_baro']
X = df_sampled.drop(columns=['pr_baro', 'time'])

# Drop rows with NaN values in y
y = y.dropna()
X = X.loc[y.index]

# Convert all columns to numeric, replacing non-numeric values with NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Impute missing values in X using the mean of each column
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Convert back to DataFrame
X = pd.DataFrame(X_imputed, columns=X.columns)

# Define the VAE model
class VAE(tf.keras.Model):
    def __init__(self, latent_dim, input_shape):
        super(VAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(shape=input_shape),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim + latent_dim)  # mean and logvar
        ])
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(shape=(latent_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(np.prod(input_shape)),
            layers.Reshape(input_shape)
        ])
        self.predictor = tf.keras.Sequential([
            layers.InputLayer(shape=(latent_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)  # Predict pr_baro
        ])

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def predict(self, z):
        return self.predictor(z)

# Define the loss function
mse_loss = tf.keras.losses.MeanSquaredError()

def compute_loss(model, x, y):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    prediction = model.predict(z)
    reconstruction_loss = mse_loss(x, x_logit)
    kl_divergence = -0.5 * tf.reduce_mean(logvar - tf.square(mean) - tf.exp(logvar) + 1)
    prediction_loss = mse_loss(y, prediction)
    return reconstruction_loss + kl_divergence + prediction_loss

# Training step
def train_step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Grid search for hyperparameter tuning
window_sizes = [5, 10, 15]
latent_dims = [2, 4, 8]
best_loss = float('inf')
best_params = {}

for window_size in window_sizes:
    for latent_dim in latent_dims:
        # Sliding window segmentation
        segments = []
        targets = []
        for i in range(len(X) - window_size):
            segment = X.iloc[i:i + window_size].values
            target = y.iloc[i + window_size]
            segments.append(segment)
            targets.append(target)

        segments = np.array(segments)
        targets = np.array(targets)

        # Ensure all values are numeric
        segments = segments.astype(float)

        # Normalize
        segments_min = segments.min(axis=0)
        segments_max = segments.max(axis=0)

        # Avoid division by zero
        range_ = segments_max - segments_min
        range_[range_ == 0] = 1

        segments = (segments - segments_min) / range_

        # Instantiate and compile the VAE
        vae = VAE(latent_dim, input_shape=(window_size, X.shape[1]))
        optimizer = tf.keras.optimizers.Adam(1e-4)

        # Training loop
        losses = []
        for epoch in range(50):  # Use fewer epochs for grid search
            loss = train_step(vae, segments, targets, optimizer)
            losses.append(loss.numpy())

        avg_loss = np.mean(losses)
        print(f"Window Size: {window_size}, Latent Dim: {latent_dim}, Avg Loss: {avg_loss}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = {'window_size': window_size, 'latent_dim': latent_dim}

print(f"Best Parameters: {best_params}, Best Loss: {best_loss}")

# Train the final model with the best parameters
window_size = best_params['window_size']
latent_dim = best_params['latent_dim']

# Sliding window segmentation
segments = []
targets = []
for i in range(len(X) - window_size):
    segment = X.iloc[i:i + window_size].values
    target = y.iloc[i + window_size]
    segments.append(segment)
    targets.append(target)

segments = np.array(segments)
targets = np.array(targets)

# Ensure all values are numeric
segments = segments.astype(float)

# Normalize
segments_min = segments.min(axis=0)
segments_max = segments.max(axis=0)

# Avoid division by zero
range_ = segments_max - segments_min
range_[range_ == 0] = 1

segments = (segments - segments_min) / range_

# Instantiate and compile the VAE
vae = VAE(latent_dim, input_shape=(window_size, X.shape[1]))
optimizer = tf.keras.optimizers.Adam(1e-4)

# Training loop
losses = []
for epoch in range(100):
    loss = train_step(vae, segments, targets, optimizer)
    losses.append(loss.numpy())
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

# Save the predictions to a file
predictions_df = pd.DataFrame(predictions.numpy(), columns=['Predicted'])
predictions_df['Actual'] = targets
predictions_df.to_csv('predictions.csv', index=False)




