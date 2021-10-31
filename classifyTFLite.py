#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tflite_runtime.interpreter as tflite
#import tensorflow.keras as keras

from datetime import datetime
import time




def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])


def main():

    start = time.time()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    # with tf.device('/cpu:0'):
    file_arr=[]
    results_arr=[]




    interpreter=tflite.Interpreter(model_path=args.model_name)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details  = interpreter.get_output_details()

    interpreter.allocate_tensors()

    i=0

    for x in os.listdir(args.captcha_dir):
        # load image and preprocess it
        raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
        rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
        image = numpy.array(rgb_data,dtype=numpy.float32) / 255.0
        # print(image)
        (c, h, w) = image.shape
        image = image.reshape([-1, c, h, w])
        # prediction = model.predict(image)

        interpreter.set_tensor(input_details[0]['index'], image)

        arr = [] #numpy.array([])
        interpreter.invoke()
        for k in range(6):
            output_data_tflite = interpreter.get_tensor(output_details[k]['index'])
            arr.append(output_data_tflite)

        file_arr.append(x)
        results_arr.append(decode(captcha_symbols, arr))
        i+=1


    with open(args.output, 'w') as output_file:
        for i in range(len(file_arr)):
            output_file.write(file_arr[i] + ", " + results_arr[i] + "\n")


    print("End Time =", datetime.now().strftime("%H:%M:%S"))
    print("Total Time Taken (mins): ", (time.time() - start)/60)

if __name__ == '__main__':
    main()
