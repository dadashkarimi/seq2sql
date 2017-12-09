#python2.7 src/main.py -d 200 -i 100 -o 100 -u 1 -t 15,5,5,5 -c lstm -m pointnet --stats-file stats.json --domain seq2sql -k 5 --dev-seed 0 --model-seed 0 --save-file params --train-source-file data/train.jsonl --train-db-file data/train.db --train-table-file data/train.tables.jsonl --train-data data/wikisql_train.tsv
'''python src/main.py \
	-d 200 \
	-i 100 \
	-o 100 \ 
	-p none \ 
	-u 1 \
	-t 15,5,5,5 \ 
	-c lstm \ 
	-m attention \ 
	--stats-file stats.json \ 
	--domain geoquery \ 
	-k 5 \ 
	--dev-seed 0 \ 
	--model-seed 0 \ 
	--train-data geo880/geo880_train100.tsv \ 
	--dev-data geo880/geo880_test280.tsv --save-file params'''

#python src/main.py \
MP_NUM_THREADS=10 THEANO_FLAGS=-openblas flags=-lopenblas python src/main.py \
	-d 200 \
	-i 100 \
	-o 100 \
	-p attention \
	-u 1 \
	-t 15,5,5,5 \
	-c lstm \
	-m attn2hist \
	--stats-file result/stats_gru.json \
	--domain overnight-calendar \
	-k 0 \
	--dev-seed 0 \
	--model-seed 0 \
	--train-data data/overnight/calendar_train.tsv \
	--dev-data data/overnight/calendar_test.tsv \
	--train-source-file data/wikisql/train.jsonl \
	--train-db-file data/wikisql/train.db \
	--train-table-file data/wikisql/train.tables.jsonl  
