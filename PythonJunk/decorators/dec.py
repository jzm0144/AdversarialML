
from inspect import getsource
from inspect import getfile
from time import time


def timer(func):
    def f(*args, **kwargs):
        before = time()
        rv = func(*args, **kwargs)
        after  = time()
        print('Elasped ',after-before)
        return rv
    return f

def ntimes(n):
    def inner(f):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                print("running {.__name__}".format(f))
                rv = f(*args, **kwargs)
            return rv
        return wrapper
    return inner

@ntimes(2)
def add(x, y = 10):
    return x+y
@ntimes(5)
def sub(x, y=10):
    return x-y

#sub = timer(sub)
#add = timer(add)



print('add(10)',      add(10))
print('add(20, 30)',  add(20, 30))
print('add("a","b")', add("a", "b"))
print('sub(10)',      sub(10))
print('sub(20, 30)',  sub(20, 30))

