"""
Utility functions dùng chung cho DMS Edge-AI Pipeline.
"""
import logging
import sys


def configure_win_console() -> None:
    """
    Đảm bảo console Windows không bị lỗi UnicodeEncodeError khi print tiếng Việt / emoji.
    An toàn khi gọi nhiều lần.
    """
    if sys.platform == 'win32':
        for stream in (sys.stdout, sys.stderr):
            if stream and hasattr(stream, 'reconfigure'):
                try:
                    stream.reconfigure(encoding='utf-8', errors='replace')
                except Exception:
                    pass


def restore_standard_logging() -> None:
    """
    Khắc phục bug của rknnlite làm biến dạng logging._nameToLevel trong Python.
    Đảm bảo uvicorn và standard logging không bị lỗi 'Unknown level: INFO'.
    """
    standard_levels = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET,
    }
    for name, level in standard_levels.items():
        logging._nameToLevel[name] = level
        logging._levelToName[level] = name
