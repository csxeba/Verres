import math


def ceil_to_nearest_power_of_2(number: int):
    base2log = math.log2(number)
    result = 2 ** math.ceil(base2log)
    return result
