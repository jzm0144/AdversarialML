# generator.py

# top-level syntax, function -> underscore method
# x()                        __call__

from time import sleep


def add1(x, y):
    return x + y

class Adder:
    def __call__(self, x, y):
        return x+y

add2 = Adder()

def compute():
    rv = []
    for i in range(10):
        sleep(0.5)
        rv.append(i)
    return rv

class Compute:

    def __iter__(self):
        self.last = 0
        return self

    def __next__(self):
        rv = self.last
        self.last += 1
        if self.last > 10:
            raise StopIteration()
        sleep(0.5)
        return rv



for val in Compute():
    print(val)

#compute = Compute

# for x in xs:       
#   pass

# xi = iter(xs)
# while True:
#    x = next(x1)




class Api:
    def run_this_first(self):
        first()
    def run_this_second(self):
        second()
    def run_this_last(self):
        last()

def api():
    first()
    yield
    second()
    yield
    last()
    

