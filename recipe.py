# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
import re
import string
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format("./model/GoogleNews-vectors-negative300.bin", binary=True)
vocab = model.vocab

def intodata(json_file):
	with open(json_file) as f:
		return json.load(f)

prob = intodata('prob.json')
ingrlist_ic = intodata('ingrlist_ic.json')
time = intodata('time.json')
step_cnts = intodata('step_cnts.json')
ap_time = [ t['ap_time'] for t in time if t ]
tot_time = [ t['tot_time'] for t in time if t ]
pp_steps = [ s['PREPARATION'] for s in step_cnts if s ]
tot_steps = [ s['TOTAL'] for s in step_cnts if s ]

def cos_sim(k_1, k_2):
	k_1 = [ re.sub('['+string.punctuation+']', '', w) for w in re.split(', |\s|/', k_1) ]
	k_2 = [ re.sub('['+string.punctuation+']', '', w) for w in re.split(', |\s|/', k_2) ]
	if k_1 == k_2:
		return 1.0
	vk_1 = [ w for w in k_1 if w in vocab ]
	vk_2 = [ w for w in k_2 if w in vocab ]
	if vk_1 and vk_2:
		v_1 = sum(model[w] for w in vk_1)/len(vk_1)
		v_2 = sum(model[w] for w in vk_2)/len(vk_2)
		return np.dot(v_1, v_2)/(np.linalg.norm(v_1) * np.linalg.norm(v_2))
	else:
		return 0
def get_fast_sim(dic1, dic2, main=True):
	
	for k in dic1.keys():
		if k not in dic2.keys():
			dic2[k] = 0
	for k in dic2.keys():
		if k not in dic1.keys():
			dic1[k] = 0
	
	if main:
		upper = sum([min(dic1[k], dic2[k])*np.log(prob[k]) for k in list(set(dic1).intersection(dic2))])
		lower = sum([max(dic1[k], dic2[k])*np.log(prob[k]) for k in list(set(dic1).union(dic2))])
	else:
		upper = sum([min(dic1[k], dic2[k])*np.log(1-prob[k]) for k in list(set(dic1).intersection(dic2))])
		lower = sum([max(dic1[k], dic2[k])*np.log(1-prob[k]) for k in list(set(dic1).union(dic2))])
		
	return min(upper/lower, 1) if lower != 0 else 0

def get_slow_sim(dic1, dic2, main=True):
	name_1 = list(dic1.keys())
	name_2 = list(dic2.keys())
	n = min(len(name_1), len(name_2))
	m = np.zeros(shape=(len(name_1), len(name_2)))
	sim = np.zeros(n)
	l_1 = []
	l_2 = []
	for x in range(len(name_1)):
		for y in range(len(name_2)):
			m[x][y] = cos_sim(name_1[x], name_2[y])
	for s in range(n):
		a, b = np.unravel_index(m.argmax(), m.shape)
		sim[s] = m[a, b]
		l_1.append(name_1[a])
		l_2.append(name_2[b])
		m[a] = np.zeros(len(name_2))
		m[:, b] = np.zeros(len(name_1))
	min_w = [min(dic1[l_1[i]], dic2[l_2[i]]) for i in range(n)]
	max_w = [max(dic1[l_1[i]], dic2[l_2[i]]) for i in range(n)]
	if len(name_1) == n:
		unpaired = list(set(name_2) - set(l_2))
		dic = dic2
	if len(name_2) == n:
		unpaired = list(set(name_1) - set(l_1))
		dic = dic1

	if n == 0:
		return 0
	elif main:
		log_prob = [np.log(prob[l_1[i]]*prob[l_2[i]]) for i in range(n)]
		upper = sum([min_w[i]*log_prob[i]*sim[i] for i in range(n)])
		lower = sum([max_w[i]*log_prob[i] for i in range(n)])
		lower += 2*sum([dic[k]*np.log(prob[k]) for k in unpaired])
	else:
		log_inv_prob = [np.log((1-prob[l_1[i]])*(1-prob[l_2[i]])) for i in range(n)]
		upper = sum([min_w[i]*log_inv_prob[i]*sim[i] for i in range(n)])
		lower = sum([max_w[i]*log_inv_prob[i] for i in range(n)])
		lower += 2*sum([dic[k]*np.log(1 - prob[k]) for k in unpaired])
	
	return min(upper/lower, 1) if lower != 0 else 0

class recipe():
	
	def __init__(self, data):
		assert type(data) == dict
		attrs = [ "time", "step_cnts", "ingr_weight" ]
		for attr in attrs:
			if attr in data.keys():
				setattr(self, attr, data[attr])

	def get_weight_frac(self):
		tot = sum(self.ingr_weight.values())
		return { k : v / tot if tot != 0 else 0 for k, v in self.ingr_weight.items() }

	def get_ingr_name(self):
		return list(self.ingr_weight.keys())

	def get_seasoning_ingr(self):
		weight_frac = self.get_weight_frac()
		if not self.ingr_weight or len(self.ingr_weight) == 1:
			return {}
			
		sorted_score = sorted([ ( k, -(1-v)*np.log(1-prob[k]) ) for k ,v in weight_frac.items() ], key = lambda x:-x[1])
		d = [ sorted_score[i][1] - sorted_score[i+1][1] for i in range(len(sorted_score)-1) ]
		val, idx = max( (val, idx) for (idx, val) in enumerate(d) )
		return { k[0] : weight_frac[k[0]] for k in sorted_score[0:idx+1] }
	
	def get_main_ingr(self):
		weight_frac = self.get_weight_frac()
		if not self.ingr_weight:
			return {}
		if len(self.ingr_weight) == 1:
			return weight_frac

		sorted_score = sorted([( k, -v*np.log(prob[k]) ) for k, v in weight_frac.items()], key = lambda x:-x[1])
		d = [ sorted_score[i][1]-sorted_score[i+1][1] for i in range(len(sorted_score)-1) ]
		val, idx = max( (val, idx) for (idx, val) in enumerate(d) )
		return {k[0] : weight_frac[k[0]] for k in sorted_score[0:idx + 1]}

	def get_other_ingr(self):
		return {k: v for k, v in self.get_weight_frac().items() if k not in self.get_main_ingr()}

	def seasoning_sim(self, data_2, isFast = True):
		dic1 = self.get_seasoning_ingr()
		dic2 = data_2.get_seasoning_ingr()

		if isFast:
			return get_fast_sim(dic1, dic2, main=False)
		else:
			return get_slow_sim(dic1, dic2, main=False)

	def main_sim(self, data_2, isFast = True):
		dic1 = self.get_main_ingr()
		dic2 = data_2.get_main_ingr()

		if isFast:
			return get_fast_sim(dic1, dic2)
		else:
			return get_slow_sim(dic1, dic2)

	def flavor_sim(self, data_2, isFast = True):
		dic1 = self.get_other_ingr()
		dic2 = data_2.get_other_ingr()

		if isFast:
			return get_fast_sim(dic1, dic2, main=False)
		else:
			return get_slow_sim(dic1, dic2, main=False)

	def overall_sim(self, data_2, isFast = True):
		dic1 = self.get_weight_frac()
		dic2 = data_2.get_weight_frac()

		if isFast:
			return get_fast_sim(dic1, dic2)
		else:
			return get_slow_sim(dic1, dic2)
	
	def ingr_diff(self):
		if not self.ingr_weight:
			return None
		return len(np.where(ingrlist_ic <= sum([-np.log(prob[k]) for k in self.get_ingr_name()]))[0]) / len(ingrlist_ic)
		
	def time_diff(self):
		return len([ (x, y) for x, y in zip(tot_time, ap_time) if x <= self.time["tot_time"] and y <= self.time["ap_time"] ]) / len(tot_time)
		
	def step_diff(self):
		return len([ (x, y) for x, y in zip(tot_steps, pp_steps) if x <= self.step_cnts["TOTAL"] and y <= self.step_cnts["PREPARATION"] ]) / len(tot_steps)
		
	def overall_diff(self):
		i_diff = self.ingr_diff()
		if i_diff:
			return 0.31 * self.time_diff() + 0.48 * i_diff + 0.21 * self.step_diff()
		else:
			return 0.5 * self.time_diff() + 0.5 * self.step_diff()
