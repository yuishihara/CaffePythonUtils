#!/usr/bin/env python
import scipy
import numpy as np
import lmdb
import sys
from caffe.io import caffe_pb2

def convert_to_jpeg(db_dir):
    env = lmdb.open(db_dir)
    datum = caffe_pb2.Datum()
    with env.begin() as txn:
        cursor = txn.cursor()
        for key_val,ser_str in cursor:
            datum.ParseFromString(ser_str)
            print "\nKey val: ", key_val
            print "\nLabel: ", datum.label
            rows = datum.height;
            cols = datum.width;
            img_pre = np.fromstring(datum.data,dtype=np.uint8)
            img = img_pre.reshape(rows, cols)
            file_name = str(key_val) + "_" + str(datum.label) + ".jpg"
            scipy.misc.toimage(img, cmin=0.0, cmax=255.0).save("data/mnist/jpg/" + file_name)

def main():
    db_dir = sys.argv[1]
    convert_to_jpeg(db_dir)

if __name__ == "__main__":
    main()
