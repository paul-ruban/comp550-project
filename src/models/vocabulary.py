from typing import Union, List


class Vocabulary:
    def __init__(
        self, 
        data : Union[List[str], List[List[str]]],
        pad_token : str = "[PAD]",
        bos_token : str = "[BOS]",
        eos_token : str = "[EOS]",
        mask_token : str = "[MASK]",
        unk_token : str = "[UNK]"
    ) -> None:

        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        self.unk_token = unk_token
        self.id2token = self.build_vocab(data=data)
        self.token2id = {t : i for i, t in self.id2token.items()}
    
    def build_vocab(self, data) -> None:
        data = [x.split() if isinstance(x, str) else x for x in data]
        # flatten the list and keep unique values
        tokens = {t for doc in data for t in doc}

        # first add special tokens
        vocab = [
            self.pad_token, self.bos_token, self.eos_token, self.mask_token, self.unk_token
        ]

        # make sure tokens do not contain special tokens
        # otherwise they will be duplicated
        tokens -= set(vocab)

        # add the remaining tokens, have them sorted
        vocab += sorted(list(tokens))

        # pack the vocab in a dict
        vocab = {i : t for i, t in enumerate(vocab)}

        return vocab
    
    def get_vocab(self) -> dict:
        return self.id2token

    def get_inv_vocab(self) -> dict:
        return self.token2id

    def __len__(self) -> int:
        return len(self.id2token)
    
    def __getitem__(self, i : Union[int, str]) -> Union[int, str]:
        # allows to fetch id or token baed on the type of parameter passed
        # if int is passed -> get token at index i
        # if str is passed -> get index of token i
        if isinstance(i, int):
            return self.id2token[i]
        elif isinstance(i, str):
            return self.token2id[i]
        else:
            raise ValueError("Invalid index type: must be int or str.")
    
    def get(self, i: Union[int, str], default : Union[str, int] = None) -> Union[int, str]:
        # a safer version of getitem that returns default if i is not found
        # this function is usefult for converting tokens into ids and vice verse
        if default is None:
            if isinstance(i, int):
                default = self.unk_token
            elif isinstance(i, str):
                default = self[self.unk_token]
            else:
                raise ValueError("Invalid index type: must be int or str.")
        try:
            return self[i]
        except KeyError:
            return default
    
    def save(self, path : str) -> None:
        tokens = [t for t, i in sorted(self.token2id.items(), key=lambda x: x[-1])]
        with open(path, 'w') as f:
            f.writelines(f"{t}\n" for t in tokens)
    
    def encode(self, data : List[str], add_special=False) -> List[List[int]]:
        # encode list of normalized sentences: tokens are separated by space
        ids = []
        for s in data:
            token_ids = [self.get(t) for t in s.split()]
            if add_special:
                token_ids = [self[self.bos_token]] + token_ids + [self[self.eos_token]]
            ids.append(token_ids)
        
        return ids
    
    def special_token_dict(self, pad : bool = True, bos : bool = True, eos : bool = True, mask : bool = True, unk : bool = True):
        dict = {}
        if pad:
            dict[self.pad_token] = self[self.pad_token]
        if bos:
            dict[self.bos_token] = self[self.bos_token]
        if eos:
            dict[self.eos_token] = self[self.eos_token]
        if mask:
            dict[self.mask_token] = self[self.mask_token]
        if unk:
            dict[self.unk_token] = self[self.unk_token]
        
        return dict

    def decode(self, data : List[List[int]], remove_special=False) -> List[str]:
        if remove_special:
            # remove speacial tokens, but keep UNK token
            special_ids = set(self.special_token_dict(unk=False).values())
        sentences = []
        for s in data:
            tokens = [self.get(i) for i in s]
            if remove_special:
                tokens = [t for t in tokens if t not in special_ids]
            sentence = ' '.join(tokens)
            sentences.append(sentence)
        
        return sentences


    @classmethod
    def from_file(
        cls, 
        path : str,
        pad_token : str = "[PAD]",
        bos_token : str = "[BOS]",
        eos_token : str = "[EOS]",
        mask_token : str = "[MASK]",
        unk_token : str = "[UNK]"
    ) -> None:
        with open(path, 'r') as f:
            # data must be list of lists [[...]] to be built properly
            data = [line.strip() for line in f.readlines()]
        
        return cls(
            data=data,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            unk_token=unk_token
        )

