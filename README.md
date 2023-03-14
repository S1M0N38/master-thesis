# master-thesis

This repository contains all the material related to my master thesis:
*papers/references*, *code*, *reports*, *etc...*

You can keep track of the progress made by looking at the commits.

1. Clone this repo: `git clone https://github.com/S1M0N38/master-thesis.git
   tesi-bertolotto`
2. Get repo path: `cd tesi-bertolotto && pwd`

-------------------------------------------------------------------------------

## Papers

> This directory contains the useful references and a collection of selected
> papers organized as a papis library.

1. Download [papis](https://papis.readthedocs.io/en/latest/index.html)
2. Create config file for papis `mkdir -p ~/.config/papis/ && touch
   ~/.config/papis/config`
3. Add the following lines to *~/.config/papis/config*
```ini
[tesi-bertolotto]
dir = ~/path/to/tesi-bertolotto/papers
```
4. Start Web application with `papis --library tesi-bertolotto serve`

> At the moment there is no easy way to download pdf locally. A possible
> solution is being discussed
> [here](https://github.com/papis/papis/discussions/525)

-------------------------------------------------------------------------------

## Code

> This directory contains all the code used to replicate other papers' results
> or to build new experiments.

All the code is develop and tested using `Python 3.10.9` and the library
versions defined in
[requirements.txt](https://github.com/S1M0N38/master-thesis/blob/main/requirements.txt).

If you prefer a [conda](https://docs.conda.io/en/latest/index.html) environment
you can use [conda-env.yml](https://github.com/S1M0N38/master-thesis/blob/main/conda-env.yml)

### Setting up Environment

Assuming that you have Python 3.10 installed,

1. Install dependencies with `python -m pip install -r requirements.txt`
2. Navigate inside *code* directory with `cd code`. Every command related to
   code (e.g. train, explore, etc.) must be execute from this very directory.

### File Structure
```
code
├── datasets
├── models
│  ├── ...
│  └── datetime-ModelName.pth
├── runs
│  ├── ...
│  └── datetime-ModelName
├── data.py
├── models.py
├── train.py
└── explore.py
```
- `datasets`: datasets automatically downloaded (e.g. CIFAR-10)
- `models`: models checkpoints.
- `runs`: files used by TensorBoard for model exploration
- `data.py`: preprocessing, datasets and dataloaders for train a model
- `models.py`: collection of model architectures
- `train.py`: training and validation loops
- `explore.py`: explore models in TensorBoard (architecture and features projection)

### Usage

- Train models with `python train.py --help`
```
usage: train.py [-h] [-q] -m {SimpleCNN,ResNet10} [-e EPOCHS]
                [-lr LEARNING_RATE] [-v VAL_PER_EPOCH] [-r RESUME]
                [-d {cpu,cuda}] [-t TAG]

Train model on CIFAR-10.

options:
  -h, --help            show this help message and exit
  -q, --quiet           Disable progress bar. (default: False)
  -m {SimpleCNN,ResNet10}, --model {SimpleCNN,ResNet10}
                        Model to train. (default: None)
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs. (default: 1)
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate. (default: 0.001)
  -v VAL_PER_EPOCH, --val-per-epoch VAL_PER_EPOCH
                        Validation loops per epoch. (default: 5)
  -r RESUME, --resume RESUME
                        Resume training of a model. (default: None)
  -d {cpu,cuda}, --device {cpu,cuda}
                        Device used for training. (default: cpu)
  -t TAG, --tag TAG     Add tag to model name. (default: )

Source code: https://github.com/S1M0N38/master-thesis
```

- Explore models with `python explore.py --help`
```
usage: explore.py [-h] [-c CHECKPOINT] [-d {cpu,cuda}] [-e EMBEDDING_SAMPLES]

Explore model with TensorBoard (architecture and embeddings).

options:
  -h, --help            show this help message and exit
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Model checkpoint to explore. Default last trained model.
  -d {cpu,cuda}, --device {cpu,cuda}
                        Device used for exploring. Default best device avaiable
  -e EMBEDDING_SAMPLES, --embedding-samples EMBEDDING_SAMPLES
                        Number of samples used to embedding visualization.

Source code: https://github.com/S1M0N38/master-thesis
```

- Start TensorBoard server ` tensorboard --logdir=runs`

-------------------------------------------------------------------------------

## Reports

> This directory contains all materials that will be produced that is not
> code (e.g. documents, slides, ...)

