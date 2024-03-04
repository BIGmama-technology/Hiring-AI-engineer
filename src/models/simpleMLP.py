import tensorflow as tf

# Define the MLP class
class MLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
        """
        Initialize a Multilayer Perceptron (MLP) model.

        Parameters:
        input_dim (int): The dimensionality of the input data.
        hidden_dim (int): The number of units in the hidden layer.
        output_dim (int): The dimensionality of the output (number of classes).
        dropout_prob (float): The dropout probability for regularization.
        """
        super(MLP, self).__init__()
        # Define layers
        self.fc1 = tf.keras.layers.Dense(hidden_dim)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_prob)
        self.fc2 = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, training=False):
        """
        Perform forward pass through the MLP.

        Parameters:
        inputs (tf.Tensor): The input data.
        training (bool): Whether the model is in training mode.

        Returns:
        tf.Tensor: The output predictions.
        """
        # Define forward pass
        x = self.fc1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x
