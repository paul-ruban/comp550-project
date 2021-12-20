## Experiments
* No GLU
* Embeddings are pushed through the bottleneck RNN masker module and projected back to original dimension, then masked and passed to EBRT for demasking
* Masker output is concatenated as external feature to BERT output and is fed to classifier RNN.