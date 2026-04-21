from __future__ import annotations

import numpy as np

class DualNumber:
    r: int | float | np.int64 | np.float64 #partie real
    d: int | float | np.int64 | np.float64 #partie dual

    def __init__(self, real: int | float | np.int64 | np.float64 = 1, dual: int | float | np.int64 | np.float64 = 0):
        # if type(real) not in [int,float,np.int64,np.float64,np.ndarray] or type(dual) not in [int,float,np.int64,np.float64,np.ndarray]:
        #     raise TypeError(f"Les parties réelles et duales doivent être de type numérique. La partie réelle était de type {type(real)}. La partie duale était de type {type(dual)}.")
        self.r = real
        self.d = dual

    def __str__(self):
        if self.r == 0:
           if self.d == 0: return "0"
           else: return str(self.d) + "\u03B5"
        elif self.d == 0:
           return str(self.r)
        else:
           if self.d > 0: sym = "+"
           else: sym = "-"
           return str(self.r) + sym + str(abs(self.d)) + "\u03B5"

    def __add__(self, b: DualNumber | [int,float,np.int64,np.float64]) -> DualNumber:
        if isinstance(b, DualNumber):
            return DualNumber(real=self.r + b.r, dual=self.d + b.d)
        elif type(b) in [int,float,np.int64,np.float64]:
           return DualNumber(real=self.r + b, dual=self.d)
        else:
            raise TypeError(f"l'opérande b doit avoir un type numérique. b était de type: {type(b)}")

    __radd__ = __add__

    def __neg__(self):
       return DualNumber( real = -self.r , dual = -self.d )

    def __sub__(self, b: DualNumber | [int,float,np.int64,np.float64]):
        return self + b.__neg__()

    def __rsub__(self, b: DualNumber | [int,float,np.int64,np.float64]):
        return b + self.__neg__()

    def __mul__(self, b: DualNumber | [int,float,np.int64,np.float64]):
        if isinstance(b, DualNumber):
            return DualNumber(real=self.r * b.r, dual=self.d * b.r + b.d * self.r)
        elif type(b) in [int,float,np.int64,np.float64,np.ndarray]:
            return DualNumber(real=self.r * b, dual=self.d * b)
        else:
           raise TypeError(f"l'opérande b doit avoir un type numérique. b était de type: {type(b)}")

    __rmul__ = __mul__

    def __mult_inverse(self):
        if self.r == 0:
            raise ValueError("Division par un élément non-inversible :{}".format(self))

        return DualNumber(real=1/self.r, dual=-(self.d)/(self.r*self.r))

    def __truediv__(self, b: DualNumber | [int,float,np.int64,np.float64]):
        if isinstance(b, DualNumber):
            return self * DualNumber.__mult_inverse(b)
        elif type(b) in [int, float, np.int64, np.float64, np.ndarray]:
            return DualNumber(real = self.r / b, dual = self.d / b)

    def __rtruediv__(self, b: DualNumber | [int,float,np.int64,np.float64]):
        return self.__mult_inverse() * b

    #### puissance: a^b, mais b est un réel.
    def __pow__(self, b: [int,float,np.int64,np.float64]):
        return DualNumber( real = self.r.__pow__(b) , dual = self.d*b*self.r.__pow__(b-1) )

    def sin(self : DualNumber):
        return DualNumber(np.sin(self.r), np.cos(self.r)*self.d)

    def cos(self : DualNumber):
        return DualNumber(np.cos(self.r), -np.sin(self.r)*self.d)

    def exp(self : DualNumber):
        return DualNumber(np.exp(self.r), np.exp(self.r)*self.d)

    def log(self : DualNumber):
        return DualNumber(np.log(self.r), (self.d/self.r))

