# master-thesis

This repository contains all the materials related to my master's thesis:
papers/references, code, thesis, and slides.

1. Clone this repo: `git clone https://github.com/S1M0N38/master-thesis.git`
2. Get the repo path: `cd master-thesis`

## Papers

This directory contains the useful references and a collection of selected
papers organized as a papis library.

1. Download [papis](https://papis.readthedocs.io/en/latest/index.html).
2. Create a config file for papis: `mkdir -p ~/.config/papis/ && touch
   ~/.config/papis/config`.
3. Add the following lines to `~/.config/papis/config`.

```ini
[master-thesis]
dir = ~/path/to/master-thesis/papers
```

4. Start the web application with `papis --library master-thesis serve`.

> Currently, there is no easy way to download PDFs locally. Every YAML file in
> the *papers* directory contains a `download` section, which is a string where
> you can download the PDF of the corresponding paper. A possible solution is
> being discussed [here](https://github.com/papis/papis/discussions/525).

## Code

This is a submodule pointing to the directory containing all the code for
training and testing models. The code is based on the
[template](https://github.com/S1M0N38/pytorch-template) 🔥.

## Thesis

This directory contains the TeX source file of the thesis, figures, plots,
tables, etc. (i.e., all the files required to compile the PDF but not the PDF
itself).

## Slides

Slides are produced using LaTeX + Beamer, using the [metropolis
theme](https://github.com/matze/mtheme).
