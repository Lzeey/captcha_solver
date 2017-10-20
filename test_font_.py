import os
import random
import pandas as pd
import numpy as np

import progressbar

from matplotlib import pyplot as plt

from claptcha import Claptcha
from captcha.image import ImageCaptcha
FONT_DIR = "fonts"
DATA_DIR = "font_test"

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
    #captcha_gen = ImageCaptcha(fonts=[os.path.join(FONT_DIR, f) for f in fonts])
    bar = progressbar.ProgressBar()
    num_images = 200000
    
    for font in bar(fonts[:2]):
    #for i in bar(range(num_images)):
        font = random.choice(fonts)
        
        #TODO: Generate some random string here
        text = gen_random_string(char_set, length=9)
        #text_len = len(text)
        
        c = Claptcha(text, os.path.join(FONT_DIR, font), noise=np.random.uniform(low=0.2))
        #c.write(os.path.join(DATA_DIR, str(text_len), text+'.png'))
        # Get PIL Image object
        text, image = c.image
        #plt.imshow(image)
        #plt.show()
        image.save(os.path.join(DATA_DIR, font +'.jpg'))
      
        captcha_gen = ImageCaptcha(fonts=[os.path.join(FONT_DIR, font)])
        img = captcha_gen.generate_image(text)
        img.save(os.path.join(DATA_DIR, font+'_.jpg'))
        #captcha_gen.write(text, os.path.join(DATA_DIR, font+'_.jpg'))
        
        