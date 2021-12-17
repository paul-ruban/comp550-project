## Experiments
* No GLU
* Embeddings are passed through the bottleneck RNN masker module and projected back to original dimension
* Masking is decided and the original embeddings are masked and then passed to BERT for demasking
* Masker output is concatenated as external feature to BERT output and is fed to classifier RNN.