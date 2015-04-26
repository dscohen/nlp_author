The `data` directory should contain the Slate `.anc` and `.txt` files. You can get them doing:
  1. `curl -O http://www.anc.org/OANC/OANC_GrAF.tgz`
  2. `tar -xvf OANC_GrAF.tgz OANC-GrAF/data/written_1/journal/slate/` (Untested)
  3. `mkdir data`
  4. `mv slate/*/*.anc slate/*/*.txt data`
  5. `rm -rf slate`

You may not want to delete the rest of the xml files, since they contain POS tags and other info.
Note: If arg list too long for mv, replace with:
  3. `find OANC-GrAF/data/written_1/journal/slate/*/ -name "*.anc" -exec mv "{}" data \;`
  3. `find OANC-GrAF/data/written_1/journal/slate/*/ -name "*.txt" -exec mv "{}" data \;`
