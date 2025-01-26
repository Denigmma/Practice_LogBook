import os
import pandas as pd

alfavit_EU = 'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ'
alfavit_RU = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'

possible_mail = ['GMAIL.COM', 'YAHOO.COM', 'OUTLOOK.COM', 'HOTMAIL.COM', 'WISOZK.COM', 'GOODWIN.ORG',
                 'HAND.COM', 'GLOVER.COM', 'TORPHY.COM', 'FRIESEN.COM','JOHNSON.COM','FEENEY.COM',
                 'ABSHIRE.BIZ','ABSHIRE.BIZ','JOHNSON.COM','VOLKMAN.BIZ']

address_key_words=['УЛ.','КВ.','ПР.']

# input_file = 'data/encoded_address_dataset.csv'
# output_file = 'data/decode_address_data.csv'
# language = "RU"

input_file = 'data/encoded_address_dataset.csv'
output_file = 'data/decoded_address_dataset.csv'
language = "RU"

def contains_key(string, keys):
    return any(key in string.upper() for key in keys)

def read_from_csv(input_file):
    df = pd.read_csv(input_file)
    # return df['email'].tolist()
    return df['Адрес'].tolist()

def write_to_csv(encode, decode, key):
    file_exists = os.path.isfile(output_file)
    df = pd.DataFrame([[encode, decode, key]], columns=['encode', 'decode', 'key'])
    df.to_csv(output_file, mode='a', header=not file_exists, index=False)

messages = read_from_csv(input_file)
decrypted_data = []
smeshenie_decrypted_data=[]
for p in range(len(messages)):
    for t in range(len(alfavit_RU if language =="RU" else alfavit_EU) // 2):
        smeshenie = t
        message = messages[p].upper()
        itog = ''
        lang = language
        if lang == 'RU':
            for i in message:
                if i in alfavit_RU:
                    mesto = alfavit_RU.find(i)
                    new_mesto = (mesto + smeshenie) % len(alfavit_RU)
                    itog += alfavit_RU[new_mesto]
                else:
                    itog += i
            if contains_key(itog, address_key_words):
                write_to_csv(messages[p], itog, t)
        else:
            for i in message:
                if i in alfavit_EU:
                    mesto = alfavit_EU.find(i)
                    new_mesto = (mesto + smeshenie) % len(alfavit_EU)
                    itog += alfavit_EU[new_mesto]
                else:
                    itog += i
            if contains_key(itog, possible_mail):
                write_to_csv(messages[p], itog, t)
            print(t,itog)