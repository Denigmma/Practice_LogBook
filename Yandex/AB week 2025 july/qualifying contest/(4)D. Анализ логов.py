"""
Ограничение времени: 1 секунда
Ограничение памяти: 64 МБ
Ввод: стандартный ввод или input.txt
Вывод: стандартный вывод или output.txt

Поисковая выдача — это упорядоченный список документов, показанных поисковой системой по текстовому запросу. Документы размещаются по релевантности и по максимальному количеству денег, которое документ может принести.

Вставьте минимальное количество документов на выдачу, чтобы максимизировать деньги и не ухудшить качество. Размер выдачи по запросу тоже не должен изменяться: новые документы вкидываются в выдачу, а низ удаляется. Порядок изначальных документов нельзя менять. Выведите максимальную суммарную выручку по всем запросам.

Ответ округлите до сотых.

Формат ввода:

Вам дан текстовый файл со следующими данными:

serpset — список размещенных поисковых выдач

new_documents — размещённые документы

Формат вывода:

Суммарная выручка по всем запросам.

Пример:

Ввод:
{
	"serpset": [
		{
			"query": "okna",
		 	"results": [
				{"position": 0, "url": "okna-msk.ru", "relevance": 0.95, "cost": 77},
				{"position": 1, "url": "okna-pvh.ru", "relevance": 0.95, "cost": 70},
				{"position": 2, "url": "ne-okna.ru", "relevance": 0.3, "cost": 100},
				{"position": 3, "url": "best-okna.ru", "relevance": 0.1, "cost": 0}
		]},
		{
			"query": "lego",
			"results": [
				{"position": 0, "url": "lego.ru", "relevance": 0.95, "cost": 15},
				{"position": 1, "url": "lego-mir.ru", "relevance": 0.7, "cost": 30},
				{"position": 2, "url": "disney.ru", "relevance": 0.3, "cost": 100}]
		}],
	"new_documents": [
		{"query": "weather", "url": "yandex.ru/pogoda", "relevance": 1, "cost": 0},
		{"query": "okna", "url": "yandex.ru/okna", "relevance": 1, "cost": 100},
		{"query": "lego", "url": "yandex.ru/lego", "relevance": 0.4, "cost": 10}
	]}

Вывод:
338.81

Примечания:

Качество мерим метрикой RelDCG = сумма (relevance_i / i)
Деньги мерим метрикой Revenue = сумма (cost_i / i)

Ответ для файла из примера — 338.81.
"""

import sys, json, math

def rel_dcg(rels):
    """RelDCG = sum_{i=1..L} rels[i-1] / i"""
    return sum(r / (i+1) for i, r in enumerate(rels))

def revenue(costs):
    """Revenue = sum_{i=1..L} costs[i-1] / sqrt(i)"""
    return sum(c / math.sqrt(i+1) for i, c in enumerate(costs))

def best_insertion(orig_rels, orig_costs, new_doc):
    """
    Пробуем вставить new_doc=(rel,cost) в каждый p=1..L,
    где после вставки мы возьмём:
      new_list_rels  = orig_rels[:p-1] + [rel] + orig_rels[p-1:-1]
      new_list_costs = аналогично по cost
    и проверим, что rel_dcg(new) >= rel_dcg(orig).
    Возвращаем максимум revenue(new), либо None, если ни в одной
    позиции вставка не прошла по качеству.
    """
    L = len(orig_rels)
    best_rev = None
    r_new, c_new = new_doc
    orig_dcg = rel_dcg(orig_rels)
    orig_rev = revenue(orig_costs)
    for p in range(1, L+1):
        nr = orig_rels[:p-1] + [r_new] + orig_rels[p-1:L-1]
        nc = orig_costs[:p-1] + [c_new] + orig_costs[p-1:L-1]
        if rel_dcg(nr) + 1e-12 < orig_dcg:
            continue
        rv = revenue(nc)
        if best_rev is None or rv > best_rev:
            best_rev = rv
    return best_rev, orig_rev

def main():
    data = sys.stdin.read()
    obj = json.loads(data)
    new_by_q = {}
    for nd in obj.get("new_documents", []):
        q = nd["query"]
        new_by_q.setdefault(q, []).append((nd["relevance"], nd["cost"]))

    total = 0.0
    for serp in obj.get("serpset", []):
        q = serp["query"]
        res = serp["results"]
        res.sort(key=lambda x: x["position"])
        orig_rels  = [r["relevance"] for r in res]
        orig_costs = [r["cost"]      for r in res]

        best_rev_for_this = revenue(orig_costs)
        for new_doc in new_by_q.get(q, []):
            br, orv = best_insertion(orig_rels, orig_costs, new_doc)
            if br is not None and br > best_rev_for_this + 1e-12:
                best_rev_for_this = br

        total += best_rev_for_this
    print(f"{total:.2f}")

if __name__ == "__main__":
    main()
