# Describing the SSMBA DA module
This manuscript tries to explain what each part of the DA module is doing:
- `fill_batch`: What happens if you're sentence if larger than 512? Can you eliminate the random replacement/unmasking prob, because it will make the experiment more complex.
- `F.pad`: pads with the padding token
- `torch.stack`
- They seem to inverse `toks` and `masks` at the end but I'm still not sure if this will cause a bug, it doesnt't!