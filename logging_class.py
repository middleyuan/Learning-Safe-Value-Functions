import logging
import sys

class Logger(object):

    def __init__(self, path):
        self.path = path
        logging.basicConfig(
                            level=logging.INFO,
                            format="[%(levelname)-5.5s]  %(message)s",
                            handlers=[
                                        logging.FileHandler("{0}/{1}.log".format(self.path, 'log')),
                                        logging.StreamHandler(sys.stdout)
                                     ]
                            )
    
    def start_log(self):
        logging.INFO
        logging.FileHandler("{0}/{1}.log".format(self.path, 'log'))
        logging.StreamHandler(sys.stdout)
    
    def stop_log(self):
        logging._handlers.clear()
        logging.shutdown()