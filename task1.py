import numpy as np


def calc_epsilon(cast):
    machine_epsilon = cast(1)
    counter = 0
    while cast(1) + cast(machine_epsilon) / cast(2) != cast(1):
        machine_epsilon = cast(machine_epsilon) / cast(2)
        counter += 1
    return machine_epsilon, counter


epsilon, n = calc_epsilon(np.float32)
w_float = 31 - n
e_max = 2 ** w_float - 1
e_min = 2 ** (-w_float)

print(epsilon, n, e_max, e_min, w_float)

# for double
epsilon, n = calc_epsilon(np.double)
w_double = 63 - n
e_max = 2 ** w_double - 1
e_min = 2 ** (-w_double)

print(epsilon, n, e_max, e_min, w_double)

print(1, 1 + epsilon, 1 + epsilon / 2, 1 + epsilon + epsilon / 2)
