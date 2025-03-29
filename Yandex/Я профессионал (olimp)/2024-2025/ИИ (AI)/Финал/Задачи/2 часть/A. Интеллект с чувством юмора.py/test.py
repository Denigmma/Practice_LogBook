import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment


with open('titles.json','r',encoding='utf-8') as f:
	titles_data=json.load(f)
with open('abstracts.json','r',encoding='utf-8') as f:
	abstracts_data=json.load(f)

titles_text=[item['title'] for item in titles_data]
abstracts_text=[item['abstract'] for item in abstracts_data]

vectorizer= TfidfVectorizer()
all_text=titles_text+abstracts_text
tfidf_matrix=vectorizer.fit_transform(all_text)

titles_vec=tfidf_matrix[:len(titles_text)]
abstracts_vec=tfidf_matrix[len(titles_text):]

sim_matrix=cosine_similarity(titles_vec, abstracts_vec)

pairs=[]
n_tittles=len(titles_text)
n_abstract=len(abstracts_text)
for i in range(n_tittles):
	for j in range(n_abstract):
		pairs.append((i,j,sim_matrix[i,j]))

pairs.sort(key=lambda x: x[2],reverse=True)

assigment_titles=set()
assigment_abstract=set()
assigment={}

for titles_idx, abstracts_idx, _ in pairs:
	if titles_idx not in assigment_titles and abstracts_idx not in assigment_abstract:
		assigment[titles_idx]=abstracts_idx
		assigment_titles.add(titles_idx)
		assigment_abstract.add(abstracts_idx)

for i in range(n_tittles):
	if i not in assigment:
		sorted_ind=np.argsort(-sim_matrix[i])
		for j in sorted_ind:
			if j not in assigment_abstract:
				assigment[i]=j
				assigment_abstract.add(j)
				break

output=[]
for i in range(n_tittles):
	abstract_assigment=assigment[i]
	output.append({"abstract_idx":abstracts_data[abstract_assigment]["abstract_idx"],"title":titles_data[i]["title"]})

with open('answ/titles1.json', 'w', encoding='utf-8') as f:
	json.dump(output,f,ensure_ascii=False, indent=2)