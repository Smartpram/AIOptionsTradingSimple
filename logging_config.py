import logging

def configure_logging(log_file='options_trading.log'):
    """Configure logging for the application."""
    
    # Create a logger
    logger = logging.getLogger('options_trading')
    logger.setLevel(logging.DEBUG)

    # Create file handler for logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
