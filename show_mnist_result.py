#!/usr/bin/env python
import numpy
import sys

def show_mnist_result(file):
    print numpy.load(file)

def main():
    file = sys.argv[1]
    show_mnist_result(file)

if __name__ == "__main__":
    main()
