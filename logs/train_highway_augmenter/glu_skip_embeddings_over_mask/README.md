## Experiments
* GLU
* Embeddings are passed through the bottleneck RNN masker module and projected back to original dimension
* Masking is decided and the original embeddings are masked and then passed to BERT for demasking
* Masker mebeddings and BERT embeddings are fed to GLU
* The masker output is concatenated as external feature to the gated output and is fed to classifier RNN.