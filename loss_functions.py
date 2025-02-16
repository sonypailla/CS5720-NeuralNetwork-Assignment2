import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Define true values and model predictions
y_true = np.array([0, 1, 1, 0])  # Example true values for binary classification
y_pred = np.array([0.1, 0.9, 0.8, 0.2])  # Example model predictions

# Step 2: Compute Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE)
# Convert y_true and y_pred to tensors
y_true_tensor = tf.constant(y_true, dtype=tf.float32)
y_pred_tensor = tf.constant(y_pred, dtype=tf.float32)

# MSE Loss
mse_loss = tf.reduce_mean(tf.square(y_true_tensor - y_pred_tensor))
print(f"Mean Squared Error (MSE): {mse_loss.numpy()}")

# Categorical Cross-Entropy Loss
# We need to use softmax to simulate categorical output (for example, in multi-class classification).
y_true_cce = np.array([[0, 1], [0, 1], [1, 0], [0, 1]])  # One-hot encoded labels for CCE

# Categorical Cross-Entropy calculation
cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
cce_value = cce_loss(y_true_cce, y_pred_tensor)
print(f"Categorical Cross-Entropy (CCE): {cce_value.numpy()}")

# Step 3: Modify predictions slightly and check how loss values change
y_pred_modified = np.array([0.3, 0.7, 0.6, 0.5])  # Modified predictions
y_pred_modified_tensor = tf.constant(y_pred_modified, dtype=tf.float32)

# Compute new losses for modified predictions
mse_loss_modified = tf.reduce_mean(tf.square(y_true_tensor - y_pred_modified_tensor))
cce_value_modified = cce_loss(y_true_cce, y_pred_modified_tensor)

print(f"Modified MSE Loss: {mse_loss_modified.numpy()}")
print(f"Modified CCE Loss: {cce_value_modified.numpy()}")

# Step 4: Plot loss function values using Matplotlib
losses = [mse_loss.numpy(), cce_value.numpy(), mse_loss_modified.numpy(), cce_value_modified.numpy()]
labels = ['MSE Loss (Original)', 'CCE Loss (Original)', 'MSE Loss (Modified)', 'CCE Loss (Modified)']

# Plotting
plt.bar(labels, losses, color=['blue', 'green', 'orange', 'red'])
plt.title('Comparison of Loss Functions')
plt.ylabel('Loss Value')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
