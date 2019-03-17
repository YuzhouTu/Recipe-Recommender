# -*- coding: utf-8 -*-

def get_time(l):
	time={}
	time["ap_time"] = l["preparation_time_seconds"] if l["preparation_time_seconds"] else 0
	time["pp_time"] = l["passive_preparation_time_seconds"] if l["passive_preparation_time_seconds"] else 0
	time["ck_time"] = sum([ll["duration_seconds"] for ll in l["steps"]]) if l["steps"] else 0
	time["tot_time"] = sum(time.values())
	return time
	
def get_step_cnts(l):
	step_c = {}
	if l["steps"]:
		for ll in l["steps"]:
			ty=ll["type"]
			if ty not in step_c.keys():
				step_c[ty] = 0
			step_c[ty] += 1
	
	if "PREPARATION" not in step_c.keys():
		step_c["PREPARATION"] = 0
	if "PROCESSING" not in step_c.keys():
		step_c["PROCESSING"] = 0
	step_c["TOTAL"]=sum(step_c.values())
	
	return step_c
	
def get_ingr_weight(l):
	return { ll["name_singular"]: ll["gram_quantity"] for ll in l["ingredients"] if ll["gram_quantity"] } if l["ingredients"] else {}

def renew(x):
	x["step_cnts"] = get_step_cnts(x)
	x["ingr_weight"] = get_ingr_weight(x)
	x["time"] = get_time(x)
	return x
