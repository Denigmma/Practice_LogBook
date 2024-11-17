alfavit_EU = 'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ'
alfavit_RU = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'

for t in range(len(alfavit_RU) // 2):
    smeshenie = t
    message = "Иугфцки ыу.м.76 тк.49".upper()
    itog = ''
    lang = "RU"
    if lang == 'RU':
        for i in message:
            if i in alfavit_RU:
                mesto = alfavit_RU.find(i)
                new_mesto = (mesto + smeshenie) % len(alfavit_RU)
                itog += alfavit_RU[new_mesto]
            else:
                itog += i
    else:
        for i in message:
            if i in alfavit_EU:
                mesto = alfavit_EU.find(i)
                new_mesto = (mesto + smeshenie) % len(alfavit_EU)
                itog += alfavit_EU[new_mesto]
            else:
                itog += i
    print(t, itog)
