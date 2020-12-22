from PIL import Image
import pandas as pd

def my_resize():
    my_data = pd.read_csv('C:\\Users\\Miguel\\Desktop\\Tese_machine_learning\\data.csv')
    print(len(my_data))
    for i in range(0, len(my_data)):
        try:
            name=my_data.loc[i, 'type']
            image = Image.open(f'C:\\Users\\Miguel\\Desktop\\Tese_machine_learning\\downloads\\{name}')
            new_image = image.resize((280, 280))
            new_image.save(f'C:\\Users\\Miguel\\Desktop\\Tese_machine_learning\\new_size\\{name}')
        except:
            print("erro")

my_resize()