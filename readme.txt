conda activate pytorch_env

Sample run
python data-process.py > ./data/dblp/dblp.content
python ./deep-comm.py --epochs 4500 --hidden=32 > ./output/email-e4500-h32-f.txt
