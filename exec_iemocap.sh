echo =======================
for iter in 1 2 3 4 5 6 7 8 9 10
do
echo --- $iter ---
python -u train.py --lr 0.0001 --batch-size 16 --epochs 150 --temp 1 --Dataset 'IEMOCAP'
done > sdt_iemocap.txt 2>&1 &