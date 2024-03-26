import json
import math
import random
import re
from bisect import bisect_left
from glob import glob
from itertools import islice

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
    content_lines = content.split('\n')
 
    total_match = re.search(r'\|Total\s+\|(.+)\|', content)
    latency_match = re.search(r'Latency', content)
    clock_match = re.search(r'\|ap_clk\s+\|(.+)\|', content)
 
    res_dict = {}
    if total_match:
        numbers_str = total_match.group(1).split('|')
        keys = ['BRAM', 'DSP', 'FF', 'LUT', 'URAM']
        res_dict = {key: int(num.strip()) if num.strip().isdigit() else 0 for key, num in zip(keys, numbers_str)}
    
    latency_dict = {}
    if latency_match:
        numbers_line = content.count('\n', 0, latency_match.start())
        
        numbers_str = []
        while len(numbers_str) < 2:
            numbers_line += 1
            numbers_str = re.findall(
                r'\d+\.\d+|\d+',
                content_lines[numbers_line]
            )
            
            # print(numbers_str)

        numbers_str = numbers_str[:2]
        keys = ['cycles_min', 'cycles_max']
        latency_dict.update({
            key: int(num.strip()) if num.strip().isdigit() else float(num.strip())\
                for key, num in zip(keys, numbers_str)
        })
    
    if clock_match:
        numbers_line = content.count('\n', 0, clock_match.start())
        
        numbers_str = numbers_str = re.findall(
            r'\d+\.\d+|\d+',
            content_lines[numbers_line]
        )[:2]
        keys = ['target_clock', 'estimated_clock']
        latency_dict.update({
            key: int(num.strip()) if num.strip().isdigit() else float(num.strip())\
                for key, num in zip(keys, numbers_str)
        })
    
    return res_dict, latency_dict

def get_count_from_files(name_pattern):
    count = 0
    
    json_paths = glob(name_pattern)
    for path in json_paths:
        with open(path, 'r') as json_file:
            json_data = json.load(json_file)
            count += len(json_data)

    return count

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

def sanitize_json(filename):
    json_data = []
    with open(filename, 'r') as json_file:
        json_data = json.load(json_file)

    for model in json_data:
        latency_report = model['latency_report']

def rnd_permutations(iterable, n_samples, start=2, max_perm=100000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    iterable = list(iterable)
    s = len(iterable)
    max_perms = min(math.factorial(s) + 1, max_perm)
    rnd_indices = rng.choice(
        range(start, max_perms),
        size=min(n_samples, max_perms - start),
        replace=False
    ) - 1
    
    res = []
    for idx in rnd_indices:
        tmp = []
        for x in range(s-1, -1, -1):
            f = math.factorial(x)
            d = idx // f
            idx -= d * f
            tmp.append(iterable[d])
            # del(iterable[d])
    
        res.append(tmp)

    return res

def limited_reservoir_sampling(iterator, n_samples, max_iters=100000):
    """
    Performs reservoir sampling on the given iterator to n_samples,
    but limits the total number of iterations.
    """
    max_iters = max(n_samples, max_iters)
    
    reservoir = list(islice(iterator, 0, n_samples))
    random.shuffle(reservoir)
    for i, item in enumerate(iterator, start=n_samples+1):
        if i >= max_iters:
            return reservoir
        
        j = random.randrange(i)
        if j < n_samples:
            reservoir[j] = item

    return reservoir

if __name__ == '__main__':
    # res_dict, latency_dict = res_from_report(
    #     './myproject_axi_csynth.rpt'
    #     # './hls4ml_prj-1/myproject_prj/solution1/syn/report/myproject_axi_csynth.rpt'
    # )
    # print(latency_dict)
    
    count = get_count_from_files('./datasets/dataset*.json')
    print(count)

    json_file = './datasets/mehdi-dataset.json'
    sanitize_json(json_file)
