'''
Учитывая массив строк strs, сгруппируйте анаграммы вместе. Вы можете вернуть ответ в любом порядке.

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]

Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Explanation:

There is no string in strs that can be rearranged to form "bat".
The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.
The strings "ate", "eat", and "tea" are anagrams as they can be rearranged to form each other.
Example 2:

Input: strs = [""]

Output: [[""]]

Example 3:

Input: strs = ["a"]

Output: [["a"]]

Ограничения:

1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i] состоит из строчных английских букв.
'''
from typing import List

# через словарь и ключ - сортированное слово
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = {}
        for word in strs:
            key=''.join(sorted(word))
            if key not in groups:
                groups[key]=[]
            groups[key].append(word)
        return list(groups.values())


# через словарь и ключ - кортеж из частоты букв
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups={}
        for word in strs:
            count=[0]*26
            for i in word:
                count[ord(i) - ord('a')] += 1
            if tuple(count) not in groups:
                groups[tuple(count)]=[]
            groups[tuple(count)].append(word)
        return list(groups.values())


solution=Solution()
strs=["eat","tea","tan","ate","nat","bat"]
print(solution.groupAnagrams(strs))


# a="abba"
# print(''.join(sorted(a))) -> aabb