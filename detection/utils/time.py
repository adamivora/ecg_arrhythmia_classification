from datetime import datetime


def timestamp():
    """
    Returns the current date and time as a string that can be used as a part of a filename.

    :return: str
    """
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
