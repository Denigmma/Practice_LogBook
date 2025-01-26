import os
import pandas as pd

alfavit_EU = 'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ'
alfavit_RU = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'

mail_key_words = ['GMAIL.COM', 'YAHOO.COM', 'OUTLOOK.COM', 'HOTMAIL.COM', 'WISOZK.COM', 'GOODWIN.ORG',
                 'HAND.COM', 'GLOVER.COM', 'TORPHY.COM', 'FRIESEN.COM', 'JOHNSON.COM', 'FEENEY.COM',
                 'ABSHIRE.BIZ', 'VOLKMAN.BIZ','MRAZ.NET']

address_key_words = ['УЛ.', 'КВ.', 'ПР.']

input_file_address = 'data/encoded_address_dataset.csv'
input_file_email = 'data/encoded_email_dataset.csv'
output_file = 'data/decoded_dataset.csv'

def contains_key(string, keys):
    return any(key in string.upper() for key in keys)

def read_from_csv(input_file, column):
    df = pd.read_csv(input_file)
    return df[column].tolist()

def write_to_csv(decoded_data):
    file_exists = os.path.isfile(output_file)
    df = pd.DataFrame(decoded_data, columns=['email', 'email_key', 'address', 'address_key'])
    df.to_csv(output_file, mode='w', header=not file_exists, index=False)

def decryptor(messages, alphabet, key_words):
    decrypted = []
    for message in messages:
        for t in range(len(alphabet) // 2):
            bias = t
            decoded_message = ''
            for char in message.upper():
                if char in alphabet:
                    old_index = alphabet.find(char)
                    new_index = (old_index + bias) % len(alphabet)
                    decoded_message += alphabet[new_index]
                else:
                    decoded_message += char
            if contains_key(decoded_message, key_words):
                decrypted.append((message, decoded_message, t))
                break
    return decrypted

def main():
    # Reading and decrypting email data
    email_data = read_from_csv(input_file_email, 'email')
    decoded_emails = decryptor(email_data, alfavit_EU, mail_key_words)

    # Reading and decrypting address data
    address_data = read_from_csv(input_file_address, 'Адрес')
    decoded_addresses = decryptor(address_data, alfavit_RU, address_key_words)

    decoded_results = []
    for i in range(max(len(decoded_emails), len(decoded_addresses))):
        email, email_decoded, email_key = decoded_emails[i] if i < len(decoded_emails) else ('', '', '')
        address, address_decoded, address_key = decoded_addresses[i] if i < len(decoded_addresses) else ('', '', '')
        decoded_results.append([email_decoded, email_key, address_decoded, address_key])

    write_to_csv(decoded_results)

main()