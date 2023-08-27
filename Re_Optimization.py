import numpy as np
import pandas as pd
class vehicle:
    def __init__(self, name, speed):
        self.name = name
        self.speed = speed
        
class ford(vehicle):
    def __init__(self, name, speed, price):
        super().__init__(name, speed)
        self.name = self.name + str('ford')
        self.price = price
    
x = ford('car', 23, 100)
