import math
import re
from bisect import bisect_left

import numpy as np


class IntRange:
    def __init__(self, min_int: int, max_int: int):
        self.min = min(min_int, max_int)
        self.max = max_int
    
    def random_in_range(self, rng=None, endpoint=True):
        if rng is None:
            rng = np.random.default_rng()
        
        return rng.integers(self.min, high=self.max, endpoint=endpoint)

class Power2Range:
    def __init__(self, min_int: int, max_int: int):
        self.min = min(min_int, max_int)
        self.max = max_int
        
        self.min_exp = max(0, int(math.log2(self.min)))
        self.max_exp = int(math.log2(self.max))
    
    def random_in_range(self, rng=None, endpoint=True):
        if rng is None:
            rng = np.random.default_rng()
        
        rnd_exp = rng.integers(self.min_exp, high=self.max_exp, endpoint=endpoint)
        return 2 ** rnd_exp

class FloatRange:
    def __init__(self, min_float: float, max_float: float):
        self.min = min(min_float, max_float)
        self.max = max_float
    
    def random_in_range(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        return rng.uniform(self.min, self.max)

def res_from_report(path):
    with open(path, 'r') as file:
        content = file.read()
 
    # Use regular expressions to find the numbers in the "Total" row
    total_match = re.search(r'\|Total\s+\|(.+)\|', content)
    latency_match = re.search(r'\bLatency', content)
 
    res_dict = {}
    if total_match:
        # Extract numbers from the matched string
        numbers_str = total_match.group(1).split('|')
        keys = ['BRAM', 'DSP', 'FF', 'LUT', 'URAM']
        res_dict = {key: int(num.strip()) if num.strip().isdigit() else 0 for key, num in zip(keys, numbers_str)}
    
    latency_dict = {}
    if latency_match:
        latency_line_number = content.count('\n', 0, latency_match.start()) + 1
        # Go down 5 lines to find numbers
        numbers_lines = content.split('\n')[latency_line_number + 5]
        # Extract numbers from the line
        numbers_str = re.findall(r'\b(\d+.\d+)\b', numbers_lines)[:4]
        keys = ['cycles_min', 'cycles_max', 'abs_min', 'abs_max']
        latency_dict = {
            key: int(num.strip()) if num.strip().isdigit() else float(num.strip())\
                for key, num in zip(keys, numbers_str)
        }
    
    return res_dict, latency_dict
    
# from hls4ml repo
def _validate_reuse_factor(n_in, n_out, rf):
    multfactor = min(n_in, rf)
    multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
    _assert = ((multiplier_limit % n_out) == 0) or (rf >= n_in)
    _assert = _assert and (((rf % n_in) == 0) or (rf < n_in))
    _assert = _assert and (((n_in * n_out) % rf) == 0)

    return _assert

def get_closest_reuse_factor(n_in, n_out, chosen_rf):
    """
    Returns closest value to chosen_rf.
    If two numbers are equally close, return the smallest number.
    """
    max_rf = n_in * n_out
    valid_reuse_factors = []
    for rf in range(1, max_rf + 1):
        _assert = _validate_reuse_factor(n_in, n_out, rf)
        if _assert:
            valid_reuse_factors.append(rf)
    valid_rf = sorted(valid_reuse_factors)
    
    pos = bisect_left(valid_rf, chosen_rf)
    if pos == 0:
        return valid_rf[0]
    if pos == len(valid_rf):
        return valid_rf[-1]
    before = valid_rf[pos - 1]
    after = valid_rf[pos]
    if after - chosen_rf < chosen_rf - before:
        return after
    else:
        return before

if __name__ == '__main__':
    res_from_report('./myproject_axi_csynth.rpt')
