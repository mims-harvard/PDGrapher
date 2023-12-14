



# Downloading and expanding processed datasets:

Download datasets [here](https://figshare.com/account/articles/24798855) and expanded manually:
* Place `torch_data.tar.gz` in `data/processed` folder and expand
* Place `splits.tar.gz`in `data` and expand

Alternatively, datasets can be downloaded and expanded using the commands:

```
mkdir processed
cd processed
wget https://figshare.com/ndownloader/files/43624557
tar -xzvf torch_data.tar.gz
cd ../
wget https://figshare.com/ndownloader/files/43632327
tar -xzvf splits.tar.gz
```