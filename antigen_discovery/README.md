# Antigen discovery for the corona virus

An MLP that predicts which of 6 HLA alleles detects a peptide of 9 amino acids, or that none of them does. It is trained on about 245,000 negative peptides and 12,000 to 25,000 positive peptides for every allele. Then it checks every 9 amino acids in a row of the Spike protein of the corona virus and prints the 3 peptides that are most likely to be detected, and by which allele.

* `antigen_discovery.py` loads the data, trains the models and checks the Spike protein
* `utils.py` the models, the training loop and the plots
* `report.pdf` the results and plots

The models:

* a simple MLP that keeps the input size of 9
* a better model: every amino acid gets a learned vector of size 8, then layers of 128 and 64 with ReLU and dropout, and a weighted loss because there are many more negatives than positives
* the better model without ReLU and dropout

## How to run

Put the data in a folder named `data` and the Spike sequence from UniProt (`P0DTC2.fasta`) next to the code, then run:

```
python antigen_discovery.py
```
