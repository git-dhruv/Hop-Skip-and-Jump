#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
flight.py: Flight Phase
"""

import sys
import logging

# Module constants
# CONSTANT_1 = "value"

# Module "global" variables
# global_var = None

class MyClass:
    """
    A simple example class with standard methods and documentation.
    """
    
    def __init__(self, attribute1, attribute2):
        """
        Initialize the MyClass instance with attribute1 and attribute2.
        """
        self.attribute1 = attribute1
        self.attribute2 = attribute2


def main()->None:    
    my_object = MyClass("value1", "value2")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    main()
