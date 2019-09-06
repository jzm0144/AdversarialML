
from inspect import getsource
from inspect import getfile
from time import time


def timer(func):
    def f(x, y=10):
        before = time()
        rv = func(x, y)
        after  = time()
        print('Elasped ',after-before)
        return rv
    return f


@timer
def add(x, y = 10):
    return x+y
@timer
def sub(x, y=10):
    return x-y

#sub = timer(sub)
#add = timer(add)



print('add(10)',      add(10))
print('add(20, 30)',  add(20, 30))
print('add("a","b")', add("a", "b"))
print('sub(10)',      sub(10))
print('sub(20, 30)',  sub(20, 30))

