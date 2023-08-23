# master-thesis

This repository contains all the materials related to my master's thesis:
papers/references, code, thesis, and slides.

1. Clone this repo: `git clone https://github.com/S1M0N38/master-thesis.git`
2. Get the repo path: `cd master-thesis`

## Papers

This directory contains the useful references and a collection of selected
papers organized as a papis library.

1. Download [papis](https://papis.readthedocs.io/en/latest/index.html).
2. Create a config file for papis: `mkdir -p ~/.config/papis/ && echo -e
   "[master-thesis]\ndir = $(pwd)/papers" > ~/.config/papis/config`
3. Start the web application with `papis --library master-thesis serve`.

> Currently, there is no easy way to download PDFs locally.  A possible
> solution is being discussed
> [here](https://github.com/papis/papis/discussions/525).

## Code

This is a git submodule pointing to the directory containing all the code for
training and testing models. The code is based on \[🔥\]
[template](https://github.com/S1M0N38/pytorch-template).

## Thesis

This directory contains the LaTeX source files, figures, plots, tables.

- [Download](https://nightly.link/S1M0N38/master-thesis/workflows/thesis/main/thesis.zip)

## Slides

Slides are produced using LaTeX + Beamer, using the [metropolis
theme](https://github.com/matze/mtheme).

- [Download](https://nightly.link/S1M0N38/master-thesis/workflows/slides/main/slides.zip)
- [Overleaf](https://www.overleaf.com/read/ndfnpkxbpgsw)
