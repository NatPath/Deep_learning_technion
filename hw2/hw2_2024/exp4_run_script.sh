python -m hw2.experiments run-exp -n exp1_4 -K 32 -L 8 -P 3 -H 100 -M "resnet"
python -m hw2.experiments run-exp -n exp1_4 -K 32 -L 16 -P 3 -H 100 -M "resnet"
python -m hw2.experiments run-exp -n exp1_4 -K 32 -L 32 -P 3 -H 100 -M "resnet"
python -m hw2.experiments run-exp -n exp1_4 -K 64 128 256 -L 2 -P 3 -H 100 -M "resnet"
python -m hw2.experiments run-exp -n exp1_4 -K 64 128 256 -L 4 -P 3 -H 100 -M "resnet"
python -m hw2.experiments run-exp -n exp1_4 -K 64 128 256 -L 8 -P 3 -H 100 -M "resnet"