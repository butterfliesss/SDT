echo =======================
for iter in 1 2 3 4 5 6 7 8 9 10
do
echo --- $iter ---
python -u train.py --lr 0.000005 --batch-size 8 --epochs 50 --temp 8 --Dataset 'MELD'
done > sdt_meld.txt 2>&1 &