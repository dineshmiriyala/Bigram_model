#this file is for text processing the input file for this would be the web scraped conversations dataset from
#https://affect.media.mit.edu/neural_chat/datasets/

#imports
import warnings
warnings.filterwarnings('ignore')
import regex
import os
import requests
import zipfile
import io

def text_processing(text):
    """returns the string removing any emojis in them and also special charecters"""
    text = regex.sub(r'\p{So}' ,'' , text)
    pattern = regex.compile('\u30c4')
    text = pattern.sub('',text)
    text = regex.sub('[^a-zA-Z\s]' , '' , text)
    text = regex.sub('\s+', ' ', text).strip()
    return text

url = 'https://affect.media.mit.edu/neural_chat/datasets/reddit_casual.zip'
req = requests.get(url)
zip = zipfile.ZipFile(io.BytesIO(req.content))

data = zip.read('reddit_casual.json').decode('utf8')

convos = []
for line in data:
    char = []
    for item in line['lines']:
        text = item['text']
    convos.append(text)
unique_convos = set(convos)
unique_convos = [*unique_convos]
final_list = []
for item in unique_convos:
    filtered = text_processing(item)
    final_list.append(filtered)

print(f'Sample data: \n{final_list[:5]}')

if not os.path.exists('reddit_convos.txt'):
    with open('reddit_convos.txt', 'w') as file:
        file.write(item)
        print("Created processed text file successfully!")
else:
    print("Processed text file exists.")


