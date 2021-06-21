import string


def int_to_string(i):
    n = len(string.ascii_uppercase)
    if i < n:
        return string.ascii_uppercase[i]
    else:
        return int_to_string((i-n)//n) + string.ascii_uppercase[i % n]
