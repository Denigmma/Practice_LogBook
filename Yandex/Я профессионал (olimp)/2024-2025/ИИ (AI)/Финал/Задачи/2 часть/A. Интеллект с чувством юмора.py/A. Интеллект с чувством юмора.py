import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment


with open('titles.json','r',encoding='utf-8') as f:
	titles_data=json.load(f)
with open('abstracts.json','r',encoding='utf-8') as f:
	abstracts_data=json.load(f)

titles_text=[item['title'] for item in titles_data]
abstracts_text=[item['abstract'] for item in abstracts_data]

model=SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

titles_emb=model.encode(titles_text,convert_to_tensor=True)
abstract_emb=model.encode(abstracts_text,convert_to_tensor=True)

titles_emb=titles_emb.cpu().numpy()
abstract_emb=abstract_emb.cpu().numpy()

sim_matrix=cosine_similarity(titles_emb, abstract_emb)

cost_matix=-sim_matrix
row_ind,col_ind=linear_sum_assignment(cost_matix)

output=[]
for titles_idx, abstracts_assigment in zip(row_ind,col_ind):
	assigned_abstract=abstracts_data[abstracts_assigment]["abstract_idx"]
	output.append({"abstract_idx":assigned_abstract,"title":titles_data[titles_idx]["title"]})

with open('answ/titles1.json', 'w', encoding='utf-8') as f:
	json.dump(output,f,ensure_ascii=False, indent=2)