import logging

MSG_FMT1 = '%(asctime)s %(filename)s %(levelname)s: %(message)s'
MSG_FMT2 = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
MSG_FMT3 = '%(asctime)s %(name)s %(filename)s - %(levelname)s: %(message)s'

logger = logging.getLogger(__name__)


def setup_logging(msg_fmt=MSG_FMT1, root_logfile=None, level=logging.INFO):
    logging.basicConfig(format=msg_fmt, level=level)
    if root_logfile is not None:
        add_handler(logging.root, root_logfile, msg_fmt=msg_fmt, level=level)


def add_handler(logger, logfile, msg_fmt=MSG_FMT1, level=logging.INFO):
    fmt = logging.Formatter(msg_fmt)
    fh = logging.FileHandler(logfile, mode='a')
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)
