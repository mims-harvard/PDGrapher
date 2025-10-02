# Union BioGRID + Menche et al. 2015 + HuRI 
import numpy as np
import networkx as nx
import pandas as pd
maxscore = 19
ref_score_list = {'Affinity Capture-MS':17,
                'Affinity Capture-Western':14,
                'Two-hybrid':1,
                'Reconstituted Complex':2,
                'Proximity Label-MS':7,
                'Co-fractionation':5,
                'Biochemical Activity':11,
                'Affinity Capture-RNA':13,
                'Co-localization':4,
                'Co-purification':6,
                'PCA':9,
                'Co-crystal Structure':18,
                'FRET':10,
                'Protein-peptide':16,
                'Affinity Capture-Luminescence':12,
                'Far Western':8,
                'Protein-RNA':3}

def read_biogrid(f, mapping):
    edges = []
    data = pd.read_csv(f, sep="\t")
    source_ = data['Entrez Gene Interactor A'].values
    target_ = data['Entrez Gene Interactor B'].values
    ref = data['Experimental System'].values
    score_ = np.array([ref_score_list[i] for i in ref])
    for i in range(len(source_)):
        source = str(source_[i])
        target = str(target_[i])
        score = score_[i]
        if source in mapping and target in mapping:
                source = mapping[source] 
                target = mapping[target]
                temp = tuple(sorted((source, target)))
                #if score == "-":
                #    score = maxscore
                temp = temp + (score,)
                edges.append(temp) 
    G = nx.Graph() 
    G.add_weighted_edges_from(edges) 
    print("BioGRID")
    #print(nx.info(G))
    for n in G.nodes:
        if "," in n: print(n)
    return G 


def read_menche(f, mapping):
    edges = []
    with open(f) as fin:
        for line in fin: 
            if line.startswith("#"): continue 
            source = line.split()[0]
            target = line.split()[1] 
            if source in mapping and target in mapping:
                source = mapping[source]
                target = mapping[target]
                temp = tuple(sorted((source, target)))
                temp = temp + (maxscore,)
                edges.append(temp)             
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    print("Menche et al. 2015")
    #print(nx.info(G))
    for n in G.nodes:
        if "," in n: print(n) 
    return G


def read_huri(f, mapping):
    edges = pd.read_csv(f, sep="\t", header=None)
    edges['weight'] = maxscore
    edges.columns = ["source", "target", "weight"]
    G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='weight')
    G = nx.relabel_nodes(G, mapping)
    for n in G.nodes:
        if "," in n: print(n)
    nodes_to_remove = []
    for n in G.nodes:
        if n.startswith("ENS"): nodes_to_remove.append(n) 
    G.remove_nodes_from(nodes_to_remove)

    print("HuRI")
    #print(nx.info(G))
    return G


def read_mapping(f):
    entrez2hgnc = dict()
    ensembl2hgnc = dict()
    with open(f) as fin:
        for line in fin:
            hgnc_id = line.split("\t")[1].strip()
            entrez_id = line.split("\t")[2].strip()
            ensembl_id = line.split("\t")[3].strip()
            if hgnc_id != "" and entrez_id != "":
                if entrez_id in entrez2hgnc: 
                    print(entrez_id, hgnc_id, entrez2hgnc[entrez_id])  
                    if hgnc_id != entrez2hgnc[entrez_id]: hgnc_id = ",".join([entrez2hgnc[entrez_id], hgnc_id]) 
                entrez2hgnc[entrez_id] = hgnc_id 
            if hgnc_id != "" and ensembl_id != "":
                if ensembl_id in ensembl2hgnc: 
                    print(ensembl_id, hgnc_id, ensembl2hgnc[ensembl_id]) 
                    if hgnc_id != ensembl2hgnc[ensembl_id]: hgnc_id = ",".join([ensembl2hgnc[ensembl_id], hgnc_id]) 
                ensembl2hgnc[ensembl_id] = hgnc_id 
    print("Num mapping entrez2hgnc", len(entrez2hgnc)) 
    print("Num mapping ensembl2hgnc", len(ensembl2hgnc)) 
    return entrez2hgnc, ensembl2hgnc


def union_G(biogrid_G, menche_G, huri_G, ppi_f):
    ppi = nx.Graph() 
    ppi.add_edges_from(biogrid_G.edges(data=True))
    ppi.add_edges_from(menche_G.edges(data=True))
    ppi.add_edges_from(huri_G.edges(data=True))
    print("Overlap with PPI + BioGRID:", len(set(list(ppi.nodes)).intersection(set(list(biogrid_G.nodes))))) 
    print("Overlap with PPI + Menche:", len(set(list(ppi.nodes)).intersection(set(list(menche_G.nodes))))) 
    print("Overlap with PPI + HuRI:", len(set(list(ppi.nodes)).intersection(set(list(huri_G.nodes))))) 
    print("Overlap with BioGRID + Menche:", len(set(list(biogrid_G.nodes)).intersection(set(list(menche_G.nodes))))) 
    print("Overlap with BioGRID + HuRI:", len(set(list(biogrid_G.nodes)).intersection(set(list(huri_G.nodes))))) 
    print("Overlap with HuRI + Menche:", len(set(list(huri_G.nodes)).intersection(set(list(menche_G.nodes))))) 
    print("Overlap with BioGRID + HuRI + Menche:", len(set(list(biogrid_G.nodes)).intersection(set(list(huri_G.nodes)), set(list(menche_G.nodes))))) 
    print("Full PPI")
    #print(nx.info(ppi)) 
    nx.write_edgelist(ppi, ppi_f, data=True)
    return ppi 


def main():  
    entrez2hgnc, ensembl2hgnc = read_mapping("../../data/ppi/2022-03-PPI/hgnc2map.txt")
    biogrid_G = read_biogrid("../../data/ppi/2022-03-PPI/BIOGRID-MV-Physical-4.4.207.tab3.txt", entrez2hgnc)
    menche_G = read_menche("../../data/ppi/2022-03-PPI/DataS1_interactome.tsv", entrez2hgnc) 
    huri_G = read_huri("../../data/ppi/2022-03-PPI/HuRI.tsv", ensembl2hgnc) 
    ppi = union_G(biogrid_G, menche_G, huri_G, "../../data/ppi/2022-03-PPI/ppi_edgelist.txt")


if __name__ == "__main__":
    main()

