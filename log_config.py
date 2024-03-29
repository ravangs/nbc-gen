import logging


class Logger:
    _instance = None

    def __new__(cls, log_name, log_file, level=logging.INFO):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_name, log_file, level=logging.INFO):
        if not hasattr(self, 'initialized'):
            self.initialized = True

            self.logger = logging.getLogger(log_name)
            self.logger.setLevel(level)

            if not self.logger.handlers:
                handler = logging.FileHandler(log_file)
                handler.setLevel(level)

                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)

                self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger
