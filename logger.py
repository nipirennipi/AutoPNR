import sys

class Logger(object):
    def __init__(self, file_path, stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
