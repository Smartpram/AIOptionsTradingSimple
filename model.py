import logging
from logging_config import configure_logging
from keras.models import Sequential
from keras.layers import Dense, Dropout

logger = configure_logging()

def create_model(input_shape: tuple) -> Sequential:
    """Creates and compiles a neural network model.

    Args:
        input_shape (tuple): Shape of the input features.

    Returns:
        Sequential: Compiled Keras model.
    """
    try:
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Output layer for regression
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        logger.info("Model creation completed.")
        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return None
