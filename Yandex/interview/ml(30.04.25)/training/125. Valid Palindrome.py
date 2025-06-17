'''
Фраза является палиндромом, если после преобразования всех заглавных букв в строчные и
 удаления всех небуквенных и нецифровых символов она читается одинаково как в прямом,
  так и в обратном направлении. К буквенно-цифровым символам относятся буквы и цифры.

Учитывая строку s, верните true если она является палиндромом, или falseв противном случае.

Example 1:

Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
Example 2:

Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.
Example 3:

Input: s = " "
Output: true
Explanation: s is an empty string "" after removing non-alphanumeric characters.
Since an empty string reads the same forward and backward, it is a palindrome.

Constraints:

1 <= s.length <= 2 * 105
s consists only of printable ASCII characters.
'''

### Шпора
example="AAbbAa123*"
example=example.lower() # сделать все буквы маленькими
example=''.join(i for i in example if i.isalnum()).lower() # убрать небуквенные символы



### решенеие за O(n) по времени и за O(n) по памяти
def isPalindrome(s: str)-> bool:
    cleaned=''.join(filter(str.isalnum, s)).lower()
    if cleaned=="":
        return True
    for i in range(len(cleaned)//2):
        if cleaned[i]!=cleaned[len(cleaned)-i-1]:
            return False
    return True


### решенеие за O(n) по времени и за O(1) по памяти
def isPalindrome(s: str) -> bool:
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum(): # правда ли, что s[left] - буква
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True



# Тренировка
def isPalindrome(s: str)-> bool:
    # s=''.join(i for i in s if i.isalnum()).lower()
    # for i in range(len(s)//2):
    #     if s[i]!=s[len(s)-i-1]:
    #         return False
    # return True

    left,right=0,len(s)-1
    while left<right:
        while left < right and not s[left].isalnum():  # правда ли, что s[left] - буква
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left+=1
        right-=1
    return True



print(isPalindrome("A man, a plan, a canal: Panama"))
print(isPalindrome(("race a car")))
print(isPalindrome(""))
print(isPalindrome("0p"))