
# clips--siameseNN

Python-built Neural Network for contrastive learning experiment (Vanderbilt Summer Research 2025)



## How to Run
First things first, you need to install python, and update it to its most recent version.
On top of that, install PyTorch package so the nn(neural network) module can run correctly. Go to this link [https://pytorch.org/get-started/locally/] for more details

After going into your IDE of choice/editing the code of train_siamese_custom.py,
Go to Command Line terminal, and write

```bash
python train_siamese_custom.py
```
This will put a "siamese_pair_model.pth" in your folder. This is the model trained on (default 200) epochs.

To TEST this model, edit the test_siamesemodel.py code (you might want to change around the anchor/test vectors) and in your terminal, run 

```bash
python test_siamesemodel.py
```
for results.
