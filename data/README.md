## 1. Download required data into appropriate folders

### 1.1 Human protein-protein interaction network data -> [data/ppi/raw](data/ppi/raw) folder
- [BIOGRID-MV-Physical-3.5.186.tab3](https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-3.5.186/BIOGRID-MV-Physical-3.5.186.tab3.zip), unzip it into BIOGRID-MV-Physical-3.5.186.tab3.txt in the same folder,
- [datasets_s1-s4.zip](https://www.science.org/doi/suppl/10.1126/science.1257601/suppl_file/datasets_s1-s4.zip), unzip it and copy DataS1_interactome.tsv from data folder into raw folder,
- [HuRI.tsv](http://www.interactome-atlas.org/data/HuRI.tsv)

### 1.2 Gene expression data -> [data/lincs/raw](data/lincs/raw) folder

From [clue.io/releases/data-dashboard](https://clue.io/releases/data-dashboard) download next files:
- cellinfo_beta.txt,
- compoundinfo_beta.txt,
- geneinfo_beta.txt,
- instinfo_beta.txt,
- level3_beta_ctl_n188708x12328.gctx,
- level3_beta_trt_sh_n453175x12328.gctx,
- level3_beta_trt_xpr_n420583x12328.gctx

### 1.3 Disease-associated genes data -> [data/cosmic/raw](data/cosmic/raw)
- [CosmicCLP_MutantExport.tsv](https://cancer.sanger.ac.uk/cell_lines/archive-download#:~:text=Complete%20mutation%20data), unzip it,
- [CosmicCLP_MutantExport.tsv](https://cancer.sanger.ac.uk/cosmic/archive-download#:~:text=COSMIC%20Complete%20Mutation%20Data%20(Targeted%20Screens)), unzip it, (copy of above at another link??)
- [expert_curated_genes_cosmic.csv](https://cancer.sanger.ac.uk/cosmic/curation)

1.4 Drug targets (mentioned in paper)
- [TODO](.)

1.5 List of cancer drugs for cancer targets baseline (mentioned in paper)
- [TODO](.)

## 2. Process the data

