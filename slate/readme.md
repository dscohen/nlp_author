The `data` directory should contain the Slate `.anc` and `.txt` files. You can get them doing:
  1. `curl -O http://www.anc.org/OANC/OANC_GrAF.tgz`
  2. `tar -xvf OANC_GrAF.tgz OANC-GrAF/data/written_1/journal/slate/` (Untested)
  3. `mkdir data`
  4. `mv slate/*/*.anc slate/*/*.txt data`
  5. `rm -rf slate`

You may not want to delete the rest of the xml files, since they contain POS tags and other info.