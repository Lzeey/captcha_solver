import os
import random
import pandas as pd
import numpy as np

import progressbar

from matplotlib import pyplot as plt

from claptcha import Claptcha
from captcha.image import ImageCaptcha
FONT_DIR = "fonts"
DATA_DIR = "data_val"

img_width = 150
img_height = 60
char_set = ('01234567890'
                'abcdefghijklmnopqrstuvwxyz'
                'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                #'!@#$%^&*')
max_len = 7
min_len = 3

def gen_random_string(char_set, length=None):
    if not length:
        length = random.randint(min_len, max_len)
    s = ''.join(random.choice(char_set) for i in range(length))
    return s
    
if __name__ == "__main__":
    # Initialize Claptcha object with "Text" as text and FreeMono as font
    fonts = [f for f in os.listdir(FONT_DIR) if f.endswith('.ttf')]    
    
    #Initialize generators
    captcha_gen = ImageCaptcha(fonts=[os.path.join(FONT_DIR, f) for f in fonts])
    
    #Generate folders first
    for i in range(min_len, max_len+1):
        tgt_dir = os.path.join(DATA_DIR, str(i))
        if not os.path.exists(tgt_dir):
            os.mkdir(tgt_dir)
    
    bar = progressbar.ProgressBar()
    num_images = 20000
    print("Generating using Claptcha")
    for i in bar(range(num_images)):
        font = random.choice(fonts)
        
        #TODO: Generate some random string here
        text = gen_random_string(char_set)
        text_len = len(text)
        
        #Generator selector
        selection = random.randint(0, 1) #Number of generators
        #selection = 0
                                  
        #For clapcha
        #Generate random amount of noise...
        if selection == 0:
            c = Claptcha(text, os.path.join(FONT_DIR, font), noise=np.random.uniform(low=0.2))
            #c.write(os.path.join(DATA_DIR, str(text_len), text+'.png'))
            # Get PIL Image object
            text, image = c.image
            #plt.imshow(image)
            #plt.show()
            try:
                image.save(os.path.join(DATA_DIR, str(text_len), text+'.jpg'))
            except OSError as e:
                print(e)
                continue
        elif selection == 1:
            try:
                captcha_gen.write(text, os.path.join(DATA_DIR, str(text_len), text+'.jpg'))
            except FileNotFoundError as e:
                print(e)
                continue
            except OSError as e:
                print(e)
                continue
        else:
            raise(Exception("Unexpected generator index found"))
        
        
