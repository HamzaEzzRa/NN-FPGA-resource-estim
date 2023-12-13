import math
import re

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
 
    if total_match:
        # Extract numbers from the matched string
        numbers_str = total_match.group(1).split('|')
        keys = ['BRAM', 'DSP', 'FF', 'LUT', 'URAM']
        numbers = {key: int(num.strip()) if num.strip().isdigit() else 0 for key, num in zip(keys, numbers_str)}
        return numbers
    else:
        print("Total row not found.")
        return None
