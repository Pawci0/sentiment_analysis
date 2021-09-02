import os
import sys


class Logger(object):
    def __init__(self, logfile, log_to_terminal=True):
        self.terminal = sys.stdout if log_to_terminal else open(os.devnull, 'w')
        self.log = open(logfile, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
