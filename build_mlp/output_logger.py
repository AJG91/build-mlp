class OutputLogger:
    """
    Logs print outputs.
    If file already exists, the new outputs will
    be appended to the existing file.
    
    Parameters
    ----------
    filename : str
        Name of file.
    """
    def __init__(self, filename):
        self.file = open(filename, "a")

    def log(self, message: str) -> None:
        """Prints message and writes message to log file."""
        print(message)
        self.file.write(message + "\n")
        self.file.flush()