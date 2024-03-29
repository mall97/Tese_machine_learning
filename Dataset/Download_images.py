import time
import urllib.request
import os
from selenium import webdriver
from selenium.webdriver.common.keys import  Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import argparse
import csv

parser = argparse.ArgumentParser(description='Download of images for training')
parser.add_argument('--obj', type=str, default= None ,help='object type / thing /animal that you search?')
args = parser.parse_args()


class webscrapping:
    def my_download(self):
        DRIVER_PATH='C:\\Users\\Miguel\\chromedriver.exe'
        browser=webdriver.Chrome(executable_path=DRIVER_PATH)
        url= 'https://www.google.com/imghp?hl=pt-pt'
        browser.get(url)
        browser.implicitly_wait(5)
        search=browser.find_element_by_name('q')
        search.send_keys(args.obj)
        search.send_keys(Keys.ENTER)
        browser.implicitly_wait(5)
        value = 0
        for i in range(40):
            browser.execute_script('scrollBy("+ str(value) +",+1000);')
            value += 1000
            time.sleep(3)

        time.sleep(30)
        
        elem1 = browser.find_element_by_id('islmp')
        sub = elem1.find_elements_by_tag_name('img')

        try:
            os.mkdir('C:\\Users\\Miguel\\Desktop\\Tese_machine_learning\\downloads')
        except FileExistsError:
            pass


        count = 0
        for i in sub:
            src = i.get_attribute('src')
            try:
                if src != None:
                    src  = str(src)
                    print(src)
                    count+=1
                    urllib.request.urlretrieve(src, os.path.join('C:\\Users\\Miguel\\Desktop\\Tese_machine_learning\\downloads', args.obj.split('_')[0]+str(count)+'.jpg'))
                else:
                    raise TypeError
            except TypeError:
                print('fail')

    def create_csv_file(self):
        entries = os.listdir('C:\\Users\\Miguel\\Desktop\\Tese_machine_learning\\new_size')
        with open('C:\\Users\\Miguel\\Desktop\\Tese_machine_learning\\data.csv', 'w', newline='') as file:
            filewriter = csv.writer(file) 
            filewriter.writerow(['type', 'class'])
            for entrie in entries:
                if entrie.find('cat')!=-1 or entrie.find('gato')!=-1:
                    filewriter.writerow([entrie, '0'])
                elif entrie.find('carro')!=-1 or entrie.find('car')!=-1:
                    filewriter.writerow([entrie, '1'])


if __name__=='__main__':
    test=webscrapping()
    #test.my_download()
    test.create_csv_file()