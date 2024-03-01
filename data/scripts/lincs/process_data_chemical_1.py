'''
Find drug targets in DrugBank, maps drugs to drugs in LINCS
'''


import pandas as pd
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
import json

import os
import csv
import gzip
import collections
import re
import io
import json
import os.path as osp
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
import pickle

outdir = '../../processed/lincs/chemical'
os.makedirs(outdir, exist_ok=True)
outdir_df = '../../processed/lincs/chemical/dataframes/'
os.makedirs(outdir_df, exist_ok=True)


def load_data(DATA_ROOT,log_handle):

	
	###Load LINCS Data
	df_lincs=pd.read_csv(os.path.join(DATA_ROOT, 'instinfo_beta.txt'), sep="\t", low_memory=False)
	df_cpmeta=pd.read_csv(os.path.join(DATA_ROOT, 'compoundinfo_beta.txt'), sep="\t", low_memory=False) 

	df_trtcp=df_lincs.loc[np.logical_and(df_lincs['pert_type'] == 'trt_cp',  df_lincs['failure_mode'].isna())]
	unique_cp=df_trtcp.pert_id.unique()

	log_handle.write('LINCS Compound Data\n------\n')
	log_handle.write('Compound treatments in inst_info:\t{}\n'.format(df_trtcp.shape[0]))
	log_handle.write('Unique pert_ids:\t{}\n'.format(df_cpmeta.pert_id.unique().shape[0]))
	log_handle.write('Unique inchi_keys:\t{}\n'.format(df_cpmeta.inchi_key.unique().shape[0]))

	###Load DrugBank Data
	DATAROOT= '../../raw/drugbank/2022-11-DrugBank/data/'
	xml_path = os.path.join(DATAROOT, 'all-full-database.xml')
	with open(xml_path) as xml_file:
		tree = ET.parse(xml_file)
	root = tree.getroot()


	ns = '{http://www.drugbank.ca}'
	inchikey_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChIKey']/{ns}value"
	inchi_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChI']/{ns}value"
	SMILES_template = "{ns}calculated-properties/{ns}property[{ns}kind='SMILES']/{ns}value"


	rows = list()
	


	for i, drug in enumerate(root):
		row = collections.OrderedDict()
		assert drug.tag == ns + 'drug'
		row['type'] = drug.get('type')
		row['drugbank_id'] = drug.findtext(ns + "drugbank-id[@primary='true']")
		row['name'] = drug.findtext(ns + "name")
		row['description'] = drug.findtext(ns + "description")
		row['groups'] = [group.text for group in
	    drug.findall("{ns}groups/{ns}group".format(ns = ns))]
		row['atc_codes'] = [code.get('code') for code in
	    drug.findall("{ns}atc-codes/{ns}atc-code".format(ns = ns))]
		
		row['categories'] = [x.findtext(ns + 'category') for x in
	    drug.findall("{ns}categories/{ns}category".format(ns = ns))]

		row['inchi'] = drug.findtext(inchi_template.format(ns = ns))
		row['inchi_key'] = drug.findtext(inchikey_template.format(ns = ns))
		row['SMILES']=drug.findtext(SMILES_template.format(ns=ns))
	
	# Add drug aliases
		aliases = {
			elem.text for elem in 
			drug.findall("{ns}international-brands/{ns}international-brand".format(ns = ns)) +
			drug.findall("{ns}synonyms/{ns}synonym[@language='English']".format(ns = ns)) +
			drug.findall("{ns}international-brands/{ns}international-brand".format(ns = ns)) +
			drug.findall("{ns}products/{ns}product/{ns}name".format(ns = ns))

		}
		aliases.add(row['name'])
		row['aliases'] = sorted(aliases)

		rows.append(row)
		
	columns = ['drugbank_id', 'name', 'type', 'groups', 'atc_codes', 'categories', 'inchi_key', 'inchi','SMILES', 'description']
	drugbank_df = pd.DataFrame.from_dict(rows)[columns]
	drugbank_slim_df = drugbank_df[
		drugbank_df.inchi.map(lambda x: x is not None) &
		drugbank_df.SMILES.map(lambda x: x is not None) 
	]
	return drugbank_slim_df, unique_cp, df_cpmeta






### Matching pert_ids to DrugBankIDs

def pert_id2inchikey(pertid, df): 
	"""Returns InChIKey of the corresponding pert_id"""
	
	
	pertid_index=df.index[df['pert_id']==pertid][0]
	return (pertid_index, df.at[pertid_index,'inchi_key'])

	
def df_pert_id2inchikey(pert_idarr, df_cpmeta):
	""" Returns a DataFrame with corresponding InChIKeys of each pert_id in pert_idarr"""

	d = {'pert_id': [], 'compoundinfo_index': [], 'inchi_key': []}
	for i in range(len(pert_idarr)): 
		d['pert_id'].append(pert_idarr[i])
		index, inchikey = pert_id2inchikey(pert_idarr[i], df_cpmeta)
		d['compoundinfo_index'].append(index)
		d['inchi_key'].append(inchikey)
	inchikey_df=pd.DataFrame(data=d)
	return inchikey_df


def df_DrugBankCol_inchi(df,drugbank_slim_df):
	"""Adds a column to df with InChIKeys mapped to DrugBank IDs from drugbank_canSmiles_df"""
	in_drugbank = set(list(drugbank_slim_df['inchi_key']))
	assert 'inchi_key' in df.columns.values
	for i in range(df.shape[0]):
		inchikey = df['inchi_key'][i]
		if type(inchikey) ==float:
			df.at[i,"DrugBank_ID"] = "None"
		else: 
			if inchikey not in in_drugbank: 
				df.at[i,"DrugBank_ID"] = "Not in DrugBank"
			else:
				ik_index=drugbank_slim_df.index[drugbank_slim_df['inchi_key']==inchikey][0]
				x= (ik_index, drugbank_slim_df.at[ik_index,'drugbank_id'])
				df.at[i,"DrugBank_ID"] = x[1]
	return df



def createMappedDF(df):
	"""Returns a copy of the df that were mapped to DrugBankIDs"""
	df_new = df.copy(deep=True)
	df_new=df_new.loc[(df_new['DrugBank_ID']!='Not in DrugBank')&(df_new['DrugBank_ID']!= "None")]
	df_new.reset_index(drop=True, inplace=True)
	return df_new





### Load DrugBank targets
def load_targets_drugbank(path):
	return pd.read_csv(path)


def summarize_drugbank_targets(mapped_DrugBankDF, df_targets):
	mapped_DrugBankDF_new = mapped_DrugBankDF.copy(deep=True)
	mapped_DrugBankDF_new['targets'] = ''
	for i in range(len(mapped_DrugBankDF)):
		dbid = mapped_DrugBankDF['DrugBank_ID'].tolist()[i]
		targets = df_targets[df_targets['DrugBank_ID']==dbid]['idd'].tolist()
		mapped_DrugBankDF_new.at[i, 'targets'] = targets
	return mapped_DrugBankDF_new




def countTargets(df): 
	#Counts number of DrugBank Targets and adds column called "num_targets"
	df['num_targets']=0
	for i in range(df.shape[0]):
		if df.notna().at[i,'targets']:
			str_list=df.at[i,'targets']
			df.at[i,'num_targets']=len(df.at[i,'targets'])
	return df
	


def target_stats(df_targets, log_handle): 
	#Calculates statistics on number of DrugBank targets and plots distribution    
	log_handle.write('DrugBank Target Stats\n------\n')
	for index,value in pd.Series.iteritems(pd.DataFrame(df_targets['num_targets']).describe()):
		log_handle.write('{}:\t{}\n'.format(index, value))
	x = list(df_targets['num_targets'])
	fig, ax1 = plt.subplots()
	ax1.hist(np.clip(x,0,30), bins=60)
	ax1.set_xlabel("# of Targets")
	ax1.set_ylabel("# of Compounds")
	ax1.set_title('Number of DrugBank Targets')
	fig.savefig(osp.join(outdir,'num_target_distribution.png'))



def main():
	DATA_ROOT = "../../raw/lincs/2022-02-LINCS_Level3/data/"
	log_handle = open(osp.join(outdir, 'log_process_data_chemical_1.txt'), 'w') 
	drugbank_slim_df, unique_cp, df_cpmeta = load_data(DATA_ROOT, log_handle)

	#Creating DataFrame mapping pert_id to InChIKeys from compoundinfo
	pert_id_inchikeyDF=df_pert_id2inchikey(unique_cp, df_cpmeta)


	#Adding the column with corresponding DrugBank IDs
	pert_id_DrugBankDF=df_DrugBankCol_inchi(pert_id_inchikeyDF,drugbank_slim_df)
	mapped_DrugBankDF=createMappedDF(pert_id_DrugBankDF)

	log_handle.write('Fraction of compounds found in DrugBank:\t{}/{}\n'.format(mapped_DrugBankDF.shape[0],pert_id_DrugBankDF.shape[0]))


	#Loads targets from DrugBank

	df_targets = load_targets_drugbank('../../processed/drugbank/targets.txt')
	df_targets.columns = ['DrugBank_ID', 'DrugBank_name', 'synonyms', 'idd', 'name', 'gene_name', 'gene_synonyms', 'identifiers', 'organism']

	#Summarizes DrugBank targets in mapped_DrugBankDF

	mapped_DrugBankDF = summarize_drugbank_targets(mapped_DrugBankDF, df_targets)
	mapped_DrugBankDF=countTargets(mapped_DrugBankDF) 


	target_stats(mapped_DrugBankDF, log_handle)
	no_targets=mapped_DrugBankDF.loc[mapped_DrugBankDF['num_targets']==0].shape[0]
	log_handle.write('Fraction of compounds without DrugBank targets:\t{}/{}\n'.format(no_targets,mapped_DrugBankDF.shape[0]))


	with open(osp.join(outdir_df,"df_targets.pickle"), 'wb') as f:
		pickle.dump(mapped_DrugBankDF, f)
	mapped_DrugBankDF.to_csv(osp.join(outdir_df,"df_targets.csv"))



if __name__ == "__main__":
	main()



