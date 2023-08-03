from bs4 import BeautifulSoup
import pandas as pd
import os

outdir = '../processed'
os.makedirs(outdir, exist_ok=True)

soup = BeautifulSoup(open("../2022-11-DrugBank/data/all-full-database.xml"),"xml")

# sep = ","
# with open('../processed/targets.txt', 'w') as f:

df = []
for drug in soup.find_all("drug"):
    drug_id = drug.find("drugbank-id").text 
    drug_name = drug.find("name").text
    targets = drug.find_all("target")
    synonyms = drug.find("synonyms")
    if synonyms is None:
        synonyms = '-'
    else:
        synonyms = synonyms.find_all("synonym")
        synonyms = "||".join([e.text for e in synonyms])
    if not targets:
        continue
    for i in targets:
        identifiers = i.find_all("external-identifier")
        identifiers = '||'.join(['|'.join([e.resource.text, e.identifier.text]) for e in identifiers])
        if i.find("id") is not None:
            idd = i.find("id").text
        else:
            idd = '-'
        if i.find("name") is not None:
            name = i.find("name").text 
        else:
            name == "-"
        if i.find("gene-name") is not None:
            gene_name = i.find("gene-name").text
        else:
            gene_name = "-"
        if i.find("organism") is not None:
            organism = i.find("organism").text
        else:
            organism = "-"
        if i.find_all('synonym') is not None:
            synonyms = '||'.join(synonym.text for synonym in i.find_all('synonym'))
        else:
            synonyms = '-'


        df.append([drug_id, drug_name, synonyms, idd, name, gene_name, synonyms, identifiers, organism])
        

df = pd.DataFrame(df, columns= ['DrugBank_ID', 'drug_name', 'drug_synonyms', 'target_id', 'target_name', 'gene_name', 'gene_synonyms', 'gene_identifiers', 'organism']).to_csv('../processed/targets.txt', sep = ',', index=False)
