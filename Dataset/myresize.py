from PIL import Image
import pandas as pd

def my_resize():
    my_data = pd.read_csv('C:\\Users\\Miguel\\Desktop\\Tese\\data.csv')
    print(len(my_data))
    for i in range(0, len(my_data)):
        try:
            image = Image.open(f'C:\\Users\\Miguel\\Desktop\\Tese\\downloads\\cat{i}.jpg')
            new_image = image.resize((280, 280))
            new_image.save(f'C:\\Users\\Miguel\\Desktop\\Tese\\new_size\\cat{i}.jpg')
        except:
            print("erro")

my_resize()