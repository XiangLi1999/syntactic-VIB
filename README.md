# VIB
Implementation of the paper: [Specializing pretrained word embeddings (for parsing) by Information Bottleneck](http://cs.jhu.edu/~jason/papers/#li-eisner-2019)

Reference
-------------------------------------------
```
@inproceedings{li-eisner-2019,
  aclid =       {D19-1276},
  author =      {Xiang Lisa Li and Jason Eisner},
  title =       {Specializing Word Embeddings (for Parsing) by
                 Information Bottleneck},
  booktitle =   {Proceedings of the 2019 Conference on Empirical
                 Methods in Natural Language Processing and 9th
                 International Joint Conference on Natural Language
                 Processing},
  pages =       {2744--2754},
  year =        {2019},
  month =       nov,
  address =     {Hong Kong},
  url =         {http://cs.jhu.edu/~jason/papers/#li-eisner-2019}
}
```

Requirements
-------------------------------------------
Roughly, the important requirements are torch==0.4.1, allennlp==0.7.1, and Python 3.6.5. Details please refer to requirement.txt


If you don't wish to run this code for languages other than English, you don't need to install ELmoForManyLangs (but might need to comment out the import at the top of conllu_handler).  


Run
-------------------------------------------
To train the discrete VIB model for English, 

```
python src/main.py --lang en --dataset_base data/UD_English/ --save_path path/to/model --out_path path/to/output --epoch 50 --mode train --type_token_reg yes --batch_size 30 --beta 0.1 --gamma -1 --seed 1 --weight_decay 0.0001 --sample_size 5   --tag_dim 128 --embedding_source elmo_1  --task VIB_discrete --cuda 1
```

To evaluate the trained discrete model, 

```
python src/main.py --lang en --dataset_base data/UD_English/ --checkpoint_path path/to/model --out_path path/to/output  --mode evaluate  --type_token_reg yes --batch_size 30 --beta 0.1 --gamma -1 --seed 1 --weight_decay 0.0001 --sample_size 5   --tag_dim 128 --embedding_source elmo_1  --task VIB_discrete --cuda 1
```

To train the continuous VIB model for English, 

```
python src/main.py --lang en --dataset_base data/UD_English/ --save_path path/to/model --out_path path/to/output --epoch 50 --mode train --type_token_reg yes --batch_size 30 --beta 0.00001 --gamma -1 --seed 1 --weight_decay 0.0001 --sample_size 5   --tag_dim 256 --embedding_source elmo_1  --task VIB_continuous --cuda 1
```

Similarly, to evaluate the trained continuous model, change from "--mode train" to "--mode evaluate"

###Quick explanations: 

- "--type_token_reg yes" is to enable the type encoder, setting it to "no" is pure VIB. 

- "--gamma -1" means setting it equal to beta, you can also tune gamma for another degree of freedom. 

- "--lang en" means the language is English. To run the code for other languages, use "--lang {en,ar,fr,hi,pt,es,ru,...}" where {} is filled by the abbreviation of different languages (see the first column of the charts below). Do not forget to modify "lang_select.py" to write the path for different languages. 

- "--embedding_source elmo_1" means using Elmo layer 1, could also swap by {elmo_0, elmo_1, elmo_2}. 

- for full explanations please use "--help" 

Additionally, 
For some baseline experiments like Iden, PCA, MLP, POS and finetune, change "--task " to be "IDEN, PCA, CLEAN, POS, FINETUNE" respectively. 


----------------------------------------------------
Some suggested hyper-param for the discrete case. 

Using batchsize = 30 gives better results than batchsize = 5.
 
For ELMo layer 1. [discrete case]

tag_dim = 128, sample_size=5, batch_size=30

| language | beta | weight-decay | 
| :---          |     :---:      |          ---: |
| ar (Arabic)   | 0.01     |  0.00001   |
| hi (Hindi)    | 0.01       | 0.00001      |
| en (English)  | 0.1       |0.00001      |
| fr (French)   | 0.1       | 0      |
| es (Spanish)  | 0.1       | 0.0001     |
| pt (Portuguese)   | 0.1       | 0.0001      |
| ru (Russian)   | 0.1       | 0.00001    |
| zh (Chinese)   | 0.1       | 0.00001     |
| it (Italian)   | 0.01      |0.0001      |


For ELMo layer 2. [discrete case]

tag_dim = 128, sample_size=5, batch_size=30

| language | beta | weight-decay | 
| :---          |     :---:      |          ---: |
| ar (Arabic)   | 0.1     |  0.00001   |
| hi (Hindi)    | 0.1     | 0      |
| en (English)  | 0.01    |0.0001      |
| fr (French)   | 0.1     | 0.00001      |
| es (Spanish)  | 0.1     | 0.0001     |
| pt (Portuguese)   | 0.1    | 0.0001      |
| ru (Russian)   | 0.1       | 0.00001    |
| zh (Chinese)   | 0.1       | 0.00001     |
| it (Italian)   | 0.1       |0.00001     |


In general, beta=0.1 is a good choice for the discrete case. 

----------------------------------------------------

For Continuous case,

For English, I set beta=1e-5, gamma=beta, weight_decay=0.0001.

In general, beta=1e-4 or 1e-5 and tag_dim=256 is a good option.







