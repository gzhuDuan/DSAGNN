
#全监督
python3 train.py --dataset Cora --lr 0.005 --weight_decay 0.0 --hidden 64 --Ko 11 --Ke 15 --Gamma 0.9 --Epsilon 0.7 --lin_lr1 0.01 --lin_weight_decay1 0.01 --lin_lr2 0.005 --lin_weight_decay2 0.005 --alpha_lin_lr 0.001 --alpha_lin_weight_decay 0.001 --optimizer_name RMSprop --dropoutClassifier1 0.3 --dropoutClassifier2 0.3 --seed 60 --epochs 1000 --split 0 --patience 200 --train_rate 0.6 --val_rate 0.2 --device 3 --runs 10

python3 train.py --dataset Citeseer --lr 0.1 --weight_decay 0.05 --hidden 256 --Ko 14 --Ke 16 --Gamma 0.3 --Epsilon 0.1 --lin_lr1 0.05 --lin_weight_decay1 0.0001 --lin_lr2 0.01 --lin_weight_decay2 0.005 --alpha_lin_lr 0.1 --alpha_lin_weight_decay 0.05 --optimizer_name RMSprop --dropoutClassifier1 0.5 --dropoutClassifier2 0.3 --seed 60 --epochs 1000 --split 0 --patience 200 --train_rate 0.6 --val_rate 0.2 --device 3 --runs 10

python3 train.py --dataset Pubmed --lr 0.05 --weight_decay 0.0005 --hidden 128 --Ko 19 --Ke 6 --Gamma 0.8 --Epsilon 0.9 --lin_lr1 0.005 --lin_weight_decay1 0.5 --lin_lr2 0.001 --lin_weight_decay2 0.05 --alpha_lin_lr 0.1 --alpha_lin_weight_decay 0.5 --optimizer_name AdamW --dropoutClassifier1 0.6 --dropoutClassifier2 0.7 --seed 60 --epochs 1000 --split 0 --patience 200 --train_rate 0.6 --val_rate 0.2 --device 3 --runs 10

python3 train.py --dataset Actor --lr 0.05 --weight_decay 0.0 --hidden 256 --Ko 16 --Ke 11 --Gamma 0.6 --Epsilon 0.1 --lin_lr1 0.005 --lin_weight_decay1 0.05 --lin_lr2 0.05 --lin_weight_decay2 0.0005 --alpha_lin_lr 0.001 --alpha_lin_weight_decay 0.0 --optimizer_name RMSprop --dropoutClassifier1 0.1 --dropoutClassifier2 0.9 --seed 60 --epochs 1000 --split 0 --patience 200 --train_rate 0.6 --val_rate 0.2 --device 3 --runs 10

python3 train.py --dataset Cornell --lr 0.05 --weight_decay 0.0005 --hidden 8 --Ko 12 --Ke 4 --Gamma 0.6 --Epsilon 0.1 --lin_lr1 0.001 --lin_weight_decay1 0.1 --lin_lr2 0.005 --lin_weight_decay2 0.005 --alpha_lin_lr 0.05 --alpha_lin_weight_decay 0.005 --optimizer_name Adam --dropoutClassifier1 0.0 --dropoutClassifier2 0.2 --seed 60 --epochs 1000 --split 0 --patience 200 --train_rate 0.6 --val_rate 0.2 --device 3 --runs 10

python3 train.py --dataset Texas --lr 0.01 --weight_decay 0.0 --hidden 64 --Ko 10 --Ke 2 --Gamma 0.3 --Epsilon 0.2 --lin_lr1 0.01 --lin_weight_decay1 0.0001 --lin_lr2 0.001 --lin_weight_decay2 0.0 --alpha_lin_lr 0.1 --alpha_lin_weight_decay 0.0001 --optimizer_name Adam --dropoutClassifier1 0.0 --dropoutClassifier2 0.3 --seed 60 --epochs 1000 --split 0 --patience 200 --train_rate 0.6 --val_rate 0.2 --device 3 --runs 10

python3 train.py --dataset Wisconsin --lr 0.01 --weight_decay 0.1 --hidden 128 --Ko 20 --Ke 17 --Gamma 0.3 --Epsilon 0.9 --lin_lr1 0.001 --lin_weight_decay1 0.1 --lin_lr2 0.1 --lin_weight_decay2 0.0001 --alpha_lin_lr 0.01 --alpha_lin_weight_decay 0.005 --optimizer_name RMSprop --dropoutClassifier1 0.3 --dropoutClassifier2 0.5 --seed 60 --epochs 1000 --split 0 --patience 200 --train_rate 0.6 --val_rate 0.2 --device 3 --runs 10


#额外的实验
python3 train.py --dataset Roman-empire --lr 0.05 --weight_decay 0.05 --hidden 256 --Ko 13 --Ke 15 --Gamma 0.5 --Epsilon 0.7 --lin_lr1 0.01 --lin_weight_decay1 0.0 --lin_lr2 0.05 --lin_weight_decay2 0.5 --alpha_lin_lr 0.001 --alpha_lin_weight_decay 0.05 --optimizer_name AdamW --dropoutClassifier1 0.0 --dropoutClassifier2 0.3 --seed 60 --epochs 1000 --split 0 --patience 200 --train_rate 0.5 --val_rate 0.25 --device 3 --runs 10

python3 train.py --dataset Amazon-ratings --lr 0.001 --weight_decay 0.0005 --hidden 128 --Ko 16 --Ke 15 --Gamma 0.7 --Epsilon 0.5 --lin_lr1 0.01 --lin_weight_decay1 0.0001 --lin_lr2 0.05 --lin_weight_decay2 0.0 --alpha_lin_lr 0.001 --alpha_lin_weight_decay 0.0001 --optimizer_name AdamW --dropoutClassifier1 0.1 --dropoutClassifier2 0.7 --seed 60 --epochs 1000 --split 0 --patience 200 --train_rate 0.5 --val_rate 0.25 --device 3 --runs 10


#半监督
python3 train.py --dataset Cora --lr 0.05 --weight_decay 0.0001 --hidden 64 --Ko 13 --Ke 17 --Gamma 0.8 --Epsilon 0.3 --lin_lr1 0.01 --lin_weight_decay1 0.005 --lin_lr2 0.001 --lin_weight_decay2 0.5 --alpha_lin_lr 0.05 --alpha_lin_weight_decay 0.1 --optimizer_name RMSprop --dropoutClassifier1 0.2 --dropoutClassifier2 0.5 --seed 60 --epochs 1000 --split 2 --patience 200 --train_rate 0.025 --val_rate 0.025 --device 3 --runs 10

python3 train.py --dataset Citeseer --lr 0.1 --weight_decay 0.0005 --hidden 256 --Ko 13 --Ke 13 --Gamma 0.5 --Epsilon 0.3 --lin_lr1 0.05 --lin_weight_decay1 0.005 --lin_lr2 0.001 --lin_weight_decay2 0.1 --alpha_lin_lr 0.05 --alpha_lin_weight_decay 0.005 --optimizer_name Adam --dropoutClassifier1 0.1 --dropoutClassifier2 0.8 --seed 60 --epochs 1000 --split 2 --patience 200 --train_rate 0.025 --val_rate 0.025 --device 3 --runs 10

python3 train.py --dataset Pubmed --lr 0.01 --weight_decay 0.0001 --hidden 256 --Ko 6 --Ke 12 --Gamma 0.9 --Epsilon 0.2 --lin_lr1 0.01 --lin_weight_decay1 0.005 --lin_lr2 0.001 --lin_weight_decay2 0.5 --alpha_lin_lr 0.05 --alpha_lin_weight_decay 0.1 --optimizer_name RMSprop --dropoutClassifier1 0.1 --dropoutClassifier2 0.3 --seed 60 --epochs 1000 --split 2 --patience 200 --train_rate 0.025 --val_rate 0.025 --device 3 --runs 10

python3 train.py --dataset Actor --lr 0.001 --weight_decay 0.0005 --hidden 128 --Ko 10 --Ke 20 --Gamma 0.7 --Epsilon 0.7 --lin_lr1 0.05 --lin_weight_decay1 0.1 --lin_lr2 0.005 --lin_weight_decay2 0.0 --alpha_lin_lr 0.05 --alpha_lin_weight_decay 0.005 --optimizer_name Adam --dropoutClassifier1 0.2 --dropoutClassifier2 0.9 --seed 60 --epochs 1000 --split 0 --patience 200 --train_rate 0.025 --val_rate 0.025 --device 3 --runs 10

python3 train.py --dataset Texas --lr 0.1 --weight_decay 0.0 --hidden 128 --Ko 16 --Ke 2 --Gamma 0.4 --Epsilon 0.6 --lin_lr1 0.01 --lin_weight_decay1 0.0 --lin_lr2 0.05 --lin_weight_decay2 0.1 --alpha_lin_lr 0.005 --alpha_lin_weight_decay 0.005 --optimizer_name RMSprop --dropoutClassifier1 0.4 --dropoutClassifier2 0.1 --seed 60 --epochs 1000 --split 0 --patience 200 --train_rate 0.025 --val_rate 0.025 --device 3 --runs 10

python3 train.py --dataset Cornell --lr 0.1 --weight_decay 0.0001 --hidden 128 --Ko 3 --Ke 2 --Gamma 0.7 --Epsilon 0.8 --lin_lr1 0.001 --lin_weight_decay1 0.0005 --lin_lr2 0.1 --lin_weight_decay2 0.05 --alpha_lin_lr 0.005 --alpha_lin_weight_decay 0.0001 --optimizer_name RMSprop --dropoutClassifier1 0.0 --dropoutClassifier2 0.1 --seed 60 --epochs 1000 --split 0 --patience 200 --train_rate 0.025 --val_rate 0.025 --device 3 --runs 10

python3 train.py --dataset Wisconsin --lr 0.05 --weight_decay 0.0 --hidden 64 --Ko 6 --Ke 18 --Gamma 0.3 --Epsilon 0.6 --lin_lr1 0.01 --lin_weight_decay1 0.0005 --lin_lr2 0.05 --lin_weight_decay2 0.1 --alpha_lin_lr 0.05 --alpha_lin_weight_decay 0.5 --optimizer_name RMSprop --dropoutClassifier1 0.1 --dropoutClassifier2 0.5 --seed 60 --epochs 1000 --split 0 --patience 200 --train_rate 0.025 --val_rate 0.025 --device 3 --runs 10


