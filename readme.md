<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/version-v1.0.0-green.svg" alt="Version">
  <img src="https://img.shields.io/badge/status-active-brightgreen.svg" alt="Status">
</p>

> Commonsense Generation Framework for Knowledge


## Installation


* Install the requirements `pip install -r requirements.txt`
* run the model

## Running a model

```

python run_cge.py --cuda --do_train --do_valid --do_test --data_path data_concept/FB15k-237_concept --save_path models/RotatE_FB15k-237_concept_0.9gama_truedata --model RotatE -d 1000 -g 12.0 -b 512 -n 4 -adv -a 1.0 -lr 5e-05 --max_steps 100000 --save_checkpoint_steps 10000 --valid_steps 50000 --log_steps 100 --test_log_steps 1000 --test_batch_size 16 --double_entity_embedding
python run_cge.py --cuda --do_train --do_valid --do_test --data_path data_concept/FB15k-237_concept --save_path models/TransE_FB15k-237_concept_0transetest --model TransE -d 1000 -g 12.0 -b 512 -n 4 -adv -a 1.0 -lr 5e-05 --max_steps 100000 --save_checkpoint_steps 10000 --valid_steps 50000 --log_steps 100 --test_log_steps 1000 --test_batch_size 16 
python run_cge.py --cuda --do_train --do_valid --do_test --data_path data_concept/FB15k-237_concept --save_path models/ComplEx_FB15k-237_concept_Complex43 --model ComplEx -d 1000 -g 4.0 -b 512 -n 2 -adv -a 1.0 -lr 5e-05 --max_steps 100000 --save_checkpoint_steps 10000 --valid_steps 50000 --log_steps 100 --test_log_steps 1000 --test_batch_size 16 --double_entity_embedding --double_relation_embedding
python run_cge.py --cuda --do_train --do_valid --do_test --data_path data_concept/FB15k-237_concept --save_path models/HAKE_FB15k-237_concept_hake-secondrepeat --model HAKE -d 1000 -g 12.0 -b 512 -n 4 -adv -a 1.0 -lr 0.0001 --max_steps 100000 --save_checkpoint_steps 10000 --valid_steps 50000 --log_steps 100 --test_log_steps 1000 --test_batch_size 16 --modulus_weight 3.5 --phase_weight 1.0

python run_cge.py --cuda --do_train --do_valid --do_test --data_path data_concept/wn18rr --save_path models/ComplEx_wn18rr_complexwn18r2r9 --model ComplEx -d 1000 -g 7.0 -b 256 -n 2 -adv -a 1.0 -lr 5e-05 --max_steps 80000 --save_checkpoint_steps 10000 --valid_steps 50000 --log_steps 100 --test_log_steps 1000 --test_batch_size 8 --double_entity_embedding --double_relation_embedding --modulus_weight 1.0 --phase_weight 0.5
python run_cge.py --cuda --do_train --do_valid --do_test --data_path data_concept/wn18rr --save_path models/RotatE_wn18rr_wn18.5agree --model RotatE -d 1000 -g 7.0 -b 256 -n 2 -adv -a 1.0 -lr 2e-05 --max_steps 100000 --save_checkpoint_steps 10000 --valid_steps 50000 --log_steps 100 --test_log_steps 1000 --test_batch_size 8 --double_entity_embedding
python run_cge.py --cuda --do_train --do_valid --do_test --data_path data_concept/wn18rr --save_path models/TransE_wn18rr_transewithall --model TransE -d 1000 -g 7.0 -b 256 -n 2 -adv -a 1.0 -lr 2e-05 --max_steps 100000 --save_checkpoint_steps 10000 --valid_steps 50000 --log_steps 100 --test_log_steps 1000 --test_batch_size 8
python run_cge.py --cuda --do_train --do_valid --do_test --data_path data_concept/wn18rr --save_path models/HAKE_wn18rr_hake-wn6repeatwithall2 --model HAKE -d 500 -g 5.0 -b 256 -n 2 -adv -a 1.0 -lr 5e-05 --max_steps 80000 --save_checkpoint_steps 10000 --valid_steps 50000 --log_steps 100 --test_log_steps 1000 --test_batch_size 8 --modulus_weight 0.5 --phase_weight 0.5

python run_cge.py --cuda --do_train --do_valid --do_test --data_path ../data_concept/YAGO3-10 --save_path ./save --model HAKE -d 500 -g 24 -b 1024 -n 256 -adv -a 1 -lr 0.0002 --max_steps 200000 --save_checkpoint_steps 10000 --valid_steps 20000 --log_steps 100 --test_log_steps 1000 --test_batch_size 1
```

## Results
The results of **CGE-HAKE** on **WN18RR**, **FB15k-237** ,**Kinships**,**UMLS** and **YAGO3-10** are as follows.
 
### FB15K-237
| | MRR |  HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| CGE-HAKE | 0.397 |0.297 | 0.437 |0.598  |

### WN18RR
| | MRR |  HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| CGE-HAKE |0.517|0.467|0.537|0.609 |

### Kinships
| | MRR |  HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| CGE-HAKE |  0.878 | 0.804 | 0.944  | 0.986 |

### UMLS
| | MRR |  HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| CGE-HAKE |  0.965|0.941|0.990| 0.999 |

### YAGO3-10
| | MRR |  HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| CGE-HAKE |  0.551|0.465|0.604| 0.705 |
