#!/usr/bin/env python3

import os
import numpy
import random
import string
import cv2
import argparse
import captcha.image
from datetime import datetime
import time


def main():

    start = time.time()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    fromSymbol=[]
    toSymbol=[]
    Rsymbols_file = open("replacedSymbols.txt", 'r')
    for l in Rsymbols_file:
        ft=l.split("\t")
        fromSymbol.append(ft[0])
        toSymbol.append(ft[1].rstrip("\n"))
    # print(fromSymbol,toSymbol)
    Rsymbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    # splitSymbols=captcha_symbols.split(" ")
    # alphanumeric=splitSymbols[0]
    # spec_chars=splitSymbols[1]

    # print(" aplhanumeric:  ",alphanumeric)
    # print(" spec chars: ",spec_chars)

    k=1
    count_11= int(args.count/args.length)
    # print(count_11)

    for m in range(args.length):

        for i in range(count_11):
            random_str = ''.join([random.choice(captcha_symbols) for j in range(k)])


            random_str_FileName=random_str

            for g in range(args.length-k):
                random_str_FileName=random_str_FileName+" "

            for i in range(len(fromSymbol)):
                random_str_FileName= random_str_FileName.replace(fromSymbol[i],toSymbol[i])
            image_path = os.path.join(args.output_dir, random_str_FileName+'.png')

            # print(random_str,"\t",random_str_FileName)

            if os.path.exists(image_path):
                version = 1
                while os.path.exists(os.path.join(args.output_dir, random_str_FileName + '_' + str(version) + '.png')):
                    version += 1
                image_path = os.path.join(args.output_dir, random_str_FileName + '_' + str(version) + '.png')

            image = numpy.array(captcha_generator.generate_image(random_str))
            cv2.imwrite(image_path, image)

        k=k+1

    print("End Time =", datetime.now().strftime("%H:%M:%S"))
    print("Total Time Taken (mins): ", (time.time() - start)/60)

if __name__ == '__main__':
    main()
