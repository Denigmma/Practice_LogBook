# math.gcd — возвращает НОД
# math.lcm — возвращает НОК

# По закрытому ключу (p,q) генерируется открытый ключ (НОД(p,q), НОК(p,q))

import math
def func():
    x,y=map(int,input().split())

    if y%x!=0:
        print(0)
        return 0
    count=0
    ab=y//x
    for a in range(1,int(math.sqrt(ab))+1):
        if ab%a==0:
            b=ab//a
            if math.gcd(a,b)==1:
                if a==b:
                    count+=1
                else:
                    count+=2
    print(count)

func()


# import math
# def func():
#     x,y=map(int,input().split())
#
#     if y%x!=0:
#         print(0)
#         return 0
#     count=0
#     ab=y//x
#     k=1
#     divisors = []
#     while ab>=k:
#         if ab%k==0:
#             divisors.append(k)
#         k+=1
#
#     for i in range(len(divisors)):
#         for j in range(i,len(divisors)):
#             if math.gcd(divisors[i],divisors[j])==1 and divisors[i]*divisors[j]==ab:
#                 if divisors[i]==divisors[j]:
#                     count+=1
#                 else:
#                     count+=2
#     print(count)
# func()
