'''
Builds data table with relevant cancer cell lines and corresponding drugs and targets
Log how many map to targets in DrugBank
'''

import pandas as pd
import os
import os.path as osp


#Outdir
outdir ='../../processed/nci'
os.makedirs(outdir, exist_ok=True)


#Cell lines and cancer
cancer_cell_mapping = {'breast': 'MCF7',
						'lung': 'A549',
						 'prostate': 'PC3'}


#Loads data
cancer_drugs = pd.read_csv('../../raw/nci/2022-11-NCI/cancer_drugs.csv')
cancer_drugs['drug'] =[e.lower() for e in cancer_drugs['drug']]
drug_targets = pd.read_csv('../../drugbank/processed/targets.txt', sep=',', low_memory=False)
drug_targets['1'] = [e.lower() for e in drug_targets['1']]
drug_targets['2'] = [str(e).lower() for e in drug_targets['2']]
drug_targets.columns = ['drug_id', 'drug_name', 'drug_synonyms', 'target_id', 'target_name', 'gene_name', 'identifiers', 'organism']


#Create dictionary of synonym-->drug
dict_syn_drug = dict()
for i in range(len(drug_targets)):
	name = drug_targets.at[i, 'drug_name']
	synonyms = drug_targets.at[i, 'drug_synonyms'].split('||') + [name]
	for s in synonyms:
		dict_syn_drug[s] = name


#Create dictionary of drug-->targets
dict_drug_targets = dict()
for i in range(len(drug_targets)):
	name = drug_targets.at[i, 'drug_name']
	targets = drug_targets[drug_targets['drug_name']==name]['gene_name'].tolist()
	dict_drug_targets[name] = targets



#For each cancer, compile targets of approved drugs
manual_mappings = {'lapatinib ditosylate': 'lapatinib',
					'osimertinib mesylate': 'osimertinib',
					'abiraterone': 'abiraterone',
					'radium 223 dichloride': 'radium ra 223 dichloride',
					'rucaparib camsylate': 'rucaparib',
					'talazoparib tosylate': 'talazoparib',
					'tepotinib hydrochloride': 'tepotinib',
					'mobocertinib succinate': 'mobocertinib',
					'dabrafenib mesylate': 'dabrafenib',
					'afatinib dimaleate': 'afatinib dimaleate',
					'trametinib dimethyl sulfoxide': 'trametinib',
					'tamoxifen citrate': 'tamoxifen',
					'abiraterone acetate': 'abiraterone',
					'erlotinib hydrochloride': 'erlotinib',
					'neratinib maleate': 'neratinib',
					'lutetium lu 177 vipivotide tetraxetan': 'lutetium lu-177 vipivotide tetraxetan',
					'capmatinib hydrochloride': 'capmatinib',
					'afatinib dimaleate' : 'afatinib',
					'capmatinib hydrochloride': 'capmatinib'}

in_drugbank = set(dict_syn_drug.keys())
table_cancer_drugs_and_targets = []


log = open('log.txt', 'w')
for cancer in list(set(cancer_drugs['cancer_type'])):
	not_mapped = []
	drugs = cancer_drugs[cancer_drugs['cancer_type'] == cancer]['drug'].tolist()
	drugs = [e.replace('\xa0', ' ').split(' (')[0] for e in drugs]
	drugs = [manual_mappings[e] if e in manual_mappings else e for e in drugs]
	for e in drugs:
		if e =='pertuzumab, trastuzumab, and hyaluronidase-zzxf':
			drugs.remove('pertuzumab, trastuzumab, and hyaluronidase-zzxf')
			drugs.append('pertuzumab')
			drugs.append('trastuzumab')
			drugs.append('hyaluronidase')
	for d in drugs:
		if d in dict_syn_drug:
			name = dict_syn_drug[d]
		else:
			not_mapped.append(d)
		targets = dict_drug_targets[name]
		targets = ','.join(targets).replace('-,','')
		table_cancer_drugs_and_targets.append([cancer_cell_mapping[cancer], name, targets])
	log.write('CANCER:\t{},\tCELL LINE:\t{}\n'.format(cancer, cancer_cell_mapping[cancer]))
	log.write('Number of approved drugs from NCI:\t{}\n'.format(len(set(drugs))))
	log.write('Mapped drugs from NCI to DrugBank:\t{}/{}\n'.format(len(set(drugs).intersection(in_drugbank)) , len(set(drugs))))
	log.write('Drugs not mapped:\n')
	for d in not_mapped:
		log.write('{}\n'.format(d))
	log.write('\n----------\n')

log.close()

df = pd.DataFrame(table_cancer_drugs_and_targets, columns=['cell_line', 'drug', 'targets'])
df.to_csv(osp.join(outdir, 'drugs_and_targets.csv'), sep='\t', index = False)

















