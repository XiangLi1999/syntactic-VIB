# VIB
Implementation of the paper: Specializing pretrained word embeddings (for parsing) by IB


Roughly, the important requirements are torch==0.4.1, allennlp==0.7.1, and Python 3.6.5. Details please refer to requirement.txt

To train the discrete VIB model for English, 

```
python src/main.py --dataset_base data/UD_English/ --checkpoint_path jobs/dis_train_local-TEST/model/job_b=5-b=0.00001-g=-1-s=1-w=0.001-s=4-m=30-s=5000-t=64-e=elmo_2 --out_path jobs/dis_train_local-TEST/output/job_b=5-b=0.00001-g=-1-s=1-w=0.001-s=4-m=30-s=5000-t=64-e=elmo_2 --epoch 50 --mode train --word_threshold 1 --projective non-projective --type_token_reg yes --batch_size 30 --beta 0.01 --gamma -1 --seed 1 --weight_decay 0.0001 --sample_size 5 --max_sent_len 30 --sent_per_epoch 5000 --tag_dim 64 --embedding_source elmo_1 --test yes  --foreign no --task VIB_discrete
```

To train the continuous VIB model for English, 

```
python src/main.py --dataset_base data/UD_English/ --checkpoint_path jobs/dis_train_local-TEST/model/job_b=5-b=0.00001-g=-1-s=1-w=0.001-s=4-m=30-s=5000-t=64-e=elmo_2 --out_path jobs/dis_train_local-TEST/output/job_b=5-b=0.00001-g=-1-s=1-w=0.001-s=4-m=30-s=5000-t=64-e=elmo_2 --epoch 50 --mode train --word_threshold 1 --projective non-projective --type_token_reg yes --batch_size 30 --beta 0.00001 --gamma -1 --seed 1 --weight_decay 0.0001 --sample_size 5 --max_sent_len 30 --sent_per_epoch 5000 --tag_dim 64 --embedding_source elmo_1 --test yes  --foreign no --task VIB_continuous
```
