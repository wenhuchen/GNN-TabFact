# GNN-TabFact

Running Command for Numeric Model:
```
  python gnn.py --model bert-base-multilingual-uncased --do_train --encoding gnn_ind --output_dir models/gnn_fp16_numeric_cross_with_attr_ind --attention cross --lr_default 5e-6
  starting the training of 0th epoch
```

# TabularBERT-TabFact
Running Command for Tabular BERT:
```
  python sparseBERT.py --model bert-base-multilingual-uncased --do_train --output_dir models/tabular_bert_titled --lr_default 5e-6 --fp16 --with_title
```
