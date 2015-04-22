# nlp_author
The `data` directory should contain the Slate `.anc` and `.txt` files. You can get them by going to http://www.anc.org/data/oanc/download/, getting the zip or tgz, and extracting the `/OANC-GrAF/data/written_1/journal/slate/` folder to this directory. Then, run
  1. `mkdir data`  
  2. `mv slate/*/*.anc slate/*/*.txt data`  
  3. `rm -rf slate`
