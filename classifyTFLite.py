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

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def decode2(characters,li):
    result = []
    for char in li:
        result.append(characters[char])
    return "".join(result)

def main():
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
    with open(args.output, 'w') as output_file:
            # json_file = open(args.model_name+'.json', 'r')
            # loaded_model_json = json_file.read()
            # json_file.close()

            # model = keras.models.model_from_json(loaded_model_json)
            # model.load_weights(args.model_name+'.h5')

            interpreter=tflite.Interpreter(model_path=args.model_name)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details  = interpreter.get_output_details()

            print("== Input details ==")
            print("shape:", input_details[0]['shape'])
            print("type:", input_details[0]['dtype'])
            print("\n== Output details ==")
            print("shape:", output_details[0]['shape'])
            print("type:", output_details[0]['dtype'])

            # interpreter.resize_tensor_input(input_details[0]['index'], (32, 64, 128, 3))
            # interpreter.resize_tensor_input(output_details[0]['index'], (32, 63))
            interpreter.allocate_tensors()

            print("== Input details ==")
            print("shape:", input_details[0]['shape'])
            print("type:", input_details[0]['dtype'])
            print("\n== Output details ==")
            print("shape:", output_details[0]['shape'])
            print("type:", output_details[0]['dtype'])

            # model.compile(loss='categorical_crossentropy',
            #               optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
            #               metrics=['accuracy'])

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

                for k in range(5):
                    print("k is "+str(k))

                    interpreter.invoke()
                    output_data_tflite = interpreter.get_tensor(output_details[k]['index'])


                    print(output_data_tflite)
                    # numpy.vstack((arr, numpy.array(output_data_tflite)))

                    arr.append(output_data_tflite)

                # labels_indices = numpy.argmax(output_data_tflite, axis=2)
                #
                # decoded_label = [decode2(x) for x in labels_indices][0]

                # print(arr)
                output_file.write(x + ", " + decode(captcha_symbols, arr) + "\n")

                print('Classified ' + x)

if __name__ == '__main__':
    main()
