This directory contains the data splits of InScript and DeScript used for supervised script parsing.
10 scenarios are considered. In case of questions contact Iza at skrjanec@coli.uni-saarland.de.

.
├── data_train	# The train splits of In- and DeScript, and the backtranslated train splits of InScript
├── data_val	# The validation splits of  InScript and DeScript
├── data_test	# The test splits of InScript and DeScript
├── inscript	# The InScript corpus, one file per scenario (no splits)
├── descript	# The DeScript corpus, one file per scenario (no splits)
├── inscript_backtranslation	# The train portion of InScript, backtranslated with Google Translate (English-French-English)
├── inscript_backtranslation_ep	# The train portion of InScript, backtranslated with Google Translate (English-French-English), with events and participants mapped from originals
├── descript_data_*(train|val|test)	# The respective splits of DeScript
├── inscript_data_*(train|val|test)	# The respective splits of InScript
├── exe_data_[source1]+[source2]+[source3]    # mixed training data
└── README.txt


Please note that the postfix of file names *might* be deceiving. For example, the files in the dir exe_data_ins+des+ins_backtrans/train_data/ have the postfix _descript (they contain DeScript stories) or _inscript (they contains original InScript stories AS WELL AS backtranslated stories; these two sources are concatenated).



