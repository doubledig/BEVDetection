from pathlib import Path
from typing import Optional, Union

import cv2
import mmengine.fileio as fileio
import numpy as np
from cv2 import (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION,
                 IMREAD_UNCHANGED)
from mmengine.utils import is_str

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED,
    'color_ignore_orientation': IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    'grayscale_ignore_orientation':
    IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE
}


def imread(img_or_path: Union[np.ndarray, str, Path],
           flag: str = 'color',
           channel_order: str = 'bgr',
           backend_args: Optional[dict] = None) -> np.ndarray:
    if isinstance(img_or_path, Path):
        img_or_path = str(img_or_path)

    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif is_str(img_or_path):
        img_bytes = fileio.get(img_or_path, backend_args=backend_args)
        return imfrombytes(img_bytes, flag, channel_order)
    else:
        raise TypeError('"img" must be a numpy array or a str or '
                        'a pathlib.Path object')


def imfrombytes(content: bytes,
                flag: str = 'color',
                channel_order: str = 'bgr') -> np.ndarray:
    img_np = np.frombuffer(content, np.uint8)
    flag = imread_flags[flag] if is_str(flag) else flag
    img = cv2.imdecode(img_np, flag)
    if flag == IMREAD_COLOR and channel_order == 'rgb':
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img
