Karpathy's [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), ported to JavaScript and running entirely in your browser. Train a real GPT from scratch — with your own data.


https://github.com/user-attachments/assets/3289fb43-2b51-4d17-9117-e661006c6052


## What is this

A visual, interactive tool that lets you train a GPT language model from scratch in your browser. Pick a dataset (or upload your own), configure the architecture, hit train, and watch it learn.

The model starts as random numbers and gradually learns the statistical patterns in your data. After training, you can generate new text that resembles the training data — fake startup names, baby names, dinosaur species, or whatever you feed it.

Everything runs on your device. No data is sent anywhere.

## How it works

This is a faithful port of Karpathy's microgpt Python script to JavaScript. The entire training loop — autograd, attention, backpropagation, Adam optimizer — runs in a Web Worker on your CPU.

The architecture is the same one behind ChatGPT, just smaller:

| | trainmyowngpt | GPT-4 |
|---|---|---|
| Parameters | ~4,000 | ~1,800,000,000,000 |
| Context | 16 characters | 128,000 tokens |
| Vocab | ~28 characters | ~100,000 subwords |
| Training | ~10 seconds | months on thousands of GPUs |

Same algorithm. Same math. Different scale.

## Datasets

Comes with four built-in datasets:

- **YC Startups** — 5,000+ Y Combinator company names
- **Baby Names** — 2,000+ popular names
- **Dinosaurs** — 1,500+ species names
- **English Words** — 10,000 common words

Or drop any `.txt` file with one entry per line.
