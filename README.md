# makemore

Rebuilding [Andrej Karpathy's makemore](https://github.com/karpathy/makemore) from scratch — all 5 parts. A deep dive into character-level language models, progressing from a simple bigram table all the way to a WaveNet-style architecture.

Each part introduces a new concept. By the end you have a solid grip on how modern language models are actually trained.

---

## Parts overview

| Folder | What it builds | Key concepts introduced |
|---|---|---|
| `part1_bigrams/` | Bigram character model | Counting tables, log-likelihood loss, one single linear layer NN |
| `part2_mlp/` | MLP language model | Embeddings, hidden layers, the Bengio et al. (2003) architecture |
| `part3_activations/` | Activation & gradient analysis | BatchNorm, dead neurons, saturated tanh, weight initialisation |
| `part4_backprop/` | Manual backpropagation | Deriving every gradient by hand through BatchNorm + cross-entropy |
| `part5_wavenet/` | WaveNet-style model | Hierarchical feature fusion, dilated causal structure |

---

## Highlight: manual backpropagation (part 4)

Part 4 is the "backprop ninja" exercise — the full backward pass is implemented manually without calling `.backward()`. This includes:

- Manual gradient for the **fused cross-entropy** (`log_softmax` + `NLL`) derived from scratch
- - Manual backward through **BatchNorm** — tracking how gradients flow through `(x - mean) / std` including the mean and variance dependencies
  - - Numerical gradient checking against PyTorch autograd at every step
   
    - This is the exercise that makes you genuinely understand what PyTorch does under the hood.
   
    - ---

    ## Standalone generators

    Beyond rebuilding the series, the architecture was applied to two independent generation tasks:

    ### `poke_gen/` — Pokémon name generator
    Trained an MLP on all Pokémon names to generate new ones in the same phonetic style.

    Sample outputs:
    ```
    bulbrin
    charmip
    squirtox
    eeveon
    mewtrix
    ```

    ### `chem_gen/` — Chemical compound name generator
    Same architecture applied to IUPAC-style chemical names.

    ---

    ## What I learned

    - Why **embedding tables** are just a differentiable lookup (the same as a one-hot × weight matrix, but cheaper)
    - - Why **BatchNorm** works: it keeps pre-activations in the linear regime of `tanh`, preventing gradient vanishing at init
      - - How the **chain rule in reverse topological order** is all backprop ever is — no magic
        - - Why **weight initialisation matters**: a bad init causes loss spikes on the first step because softmax is confidently wrong
          - - How WaveNet uses **hierarchical grouping** to process sequences efficiently
           
            - ---

            ## Project structure

            ```
            makemore/
            ├── part1_bigrams/        # bigram counting + 1-layer NN
            ├── part2_mlp/            # MLP with embeddings
            ├── part3_activations/    # BatchNorm + activation stats
            ├── part4_backprop/       # full manual backward pass
            ├── part5_wavenet/        # WaveNet-style architecture
            ├── poke_gen/             # Pokémon name generator (standalone)
            └── chem_gen/             # Chemical name generator (standalone)
            ```

            Each folder contains a self-contained Jupyter notebook.

            ---

            ## Related projects

            - [`micro_grad`](https://github.com/RaGHaV-186/micro_grad) — scalar autograd engine built from scratch; the foundation for understanding what `.backward()` actually does
           
            - ---

            ## Series reference

            Based on [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) by Andrej Karpathy. The series is the best free resource for understanding deep learning from first principles.
            
