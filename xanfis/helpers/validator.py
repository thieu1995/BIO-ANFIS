#!/usr/bin/env python
# Created by "Thieu" at 21:39, 10/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import operator
import numpy as np
from numbers import Number


SEQUENCE = (list, tuple, np.ndarray)
DIGIT = (int, np.integer)
REAL = (float, np.floating)


def is_in_bound(value, bound):
    ops = None
    if type(bound) is tuple:
        ops = operator.lt
    elif type(bound) is list:
        ops = operator.le
    if bound[0] == float("-inf") and bound[1] == float("inf"):
        return True
    elif bound[0] == float("-inf") and ops(value, bound[1]):
        return True
    elif ops(bound[0], value) and bound[1] == float("inf"):
        return True
    elif ops(bound[0], value) and ops(value, bound[1]):
        return True
    return False


def is_str_in_sequence(value: str, my_list: any):
    if type(value) == str and isinstance(my_list, SEQUENCE):
        return True if value in my_list else False
    return False


def check_int(name: str, value: None, bound=None):
    if isinstance(value, Number):
        if bound is None:
            return int(value)
        elif is_in_bound(value, bound):
            return int(value)
    bound = "" if bound is None else f"and value should be in range: {bound}"
    raise ValueError(f"'{name}' is an integer {bound}.")


def check_float(name: str, value: None, bound=None):
    if isinstance(value, Number):
        if bound is None:
            return float(value)
        elif is_in_bound(value, bound):
            return float(value)
    bound = "" if bound is None else f"and value should be in range: {bound}"
    raise ValueError(f"'{name}' is a float {bound}.")


def check_str(name: str, value: str, bound=None):
    if type(value) is str:
        if bound is None or is_str_in_sequence(value, bound):
            return value
    bound = "" if bound is None else f"and value should be one of this: {bound}"
    raise ValueError(f"'{name}' is a string {bound}.")


def check_bool(name: str, value: bool, bound=(True, False)):
    if type(value) is bool:
        if value in bound:
            return value
    bound = "" if bound is None else f"and value should be one of this: {bound}"
    raise ValueError(f"'{name}' is a boolean {bound}.")


def check_tuple_int(name: str, values: None, bounds=None):
    if isinstance(values, SEQUENCE) and len(values) > 1:
        value_flag = [isinstance(item, DIGIT) for item in values]
        if np.all(value_flag):
            if bounds is not None and len(bounds) == len(values):
                value_flag = [is_in_bound(item, bound) for item, bound in zip(values, bounds)]
                if np.all(value_flag):
                    return values
            else:
                return values
    bounds = "" if bounds is None else f"and values should be in range: {bounds}"
    raise ValueError(f"'{name}' are integer {bounds}.")


def check_tuple_float(name: str, values: tuple, bounds=None):
    if isinstance(values, SEQUENCE) and len(values) > 1:
        value_flag = [isinstance(item, Number) for item in values]
        if np.all(value_flag):
            if bounds is not None and len(bounds) == len(values):
                value_flag = [is_in_bound(item, bound) for item, bound in zip(values, bounds)]
                if np.all(value_flag):
                    return values
            else:
                return values
    bounds = "" if bounds is None else f"and values should be in range: {bounds}"
    raise ValueError(f"'{name}' are float {bounds}.")
