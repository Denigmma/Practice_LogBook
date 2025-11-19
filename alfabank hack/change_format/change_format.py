import pandas as pd
import json
import ast

df = pd.read_csv("submit.csv")

def cut_top5(x: str) -> str:
    # строка вида "[372, 114, ...]"
    lst = ast.literal_eval(x)
    top5 = lst[5:]
    return json.dumps(top5, ensure_ascii=False)

df["web_list"] = df["web_list"].apply(cut_top5)
df.to_csv("submit_top5_new.csv", index=False)

df.head()
