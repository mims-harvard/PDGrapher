## Builld PPI for PDGrapher  
1. Using hgnc2map.txt (in data/raw/ppi/2022-03-PPI) with the raw PPI files (links are listed in the same folder), run union_ppi.py to generate ppi_edgelist.txt  
2. Using geneinfo_beta.txt and ppi_edgelist.txt, run export_ppi_all_genes.py to build ppi_all_genes_edgelist.txt
