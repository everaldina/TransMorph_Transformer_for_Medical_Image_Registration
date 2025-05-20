import os
import sys

class Logger(object):
    def __init__(self, save_dir, file_name='logfile.log'):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, file_name), "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()