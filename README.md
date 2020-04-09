# GNN-TabFact
This is the state-of-the-art models on [TabFact Dataset](https://tabfact.github.io/), we leverage the idea proposed in [NumGNN](https://arxiv.org/pdf/1910.06701.pdf) into the encoding of tabular data.

- Cross-Attention Between Table and Statement to obtain the representation
- Construct the greater/less mask for the table numeric columns
- Use the the dense greater/less connection to propagate the information in each cell
- Obtain the graph repsentation obtained by the NumGNN
- Finally use it to do the two-way classification.

# Performance
We demonstrate our results as follows:
| Model     | Dev  | Test |
|-----------|------|------|
| TableBERT | 66.1 | 65.1 |
| GNN       | 72.1 | 72.2 |

# Training and Evaluating
Loading the trained GNN Model and reproduce the results:
```
CUDA_VISIBLE_DEVIES=0 python gnn.py --model bert-base-multilingual-uncased --do_test --encoding gnn --load_from models/gnn_fp16_numeric/model_ep4.pt --fp16
```

Retrain your own GNN Model for TabFact:
```
  python gnn.py --model bert-base-multilingual-uncased --do_train --encoding gnn_ind --output_dir models/gnn_fp16_numeric_cross --attention cross --lr_default 5e-6 --fp16
```



