from __future__ import annotations

import math
import time
from typing import Optional, Union

import torch
from transformers import PretrainedConfig

from petals.constants import DTYPE_MAP
from petals.utils.convert_block import QuantType

# ... existing code ... 