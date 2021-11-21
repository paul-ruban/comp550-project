import time

start_time = time.time()

compressed_text_file = open("digit_compressor/compressed/1-compressed-texts.txt", "r")
compressed_tokens = compressed_text_file.read().split(' ')
# print(tokens)
# print(len(tokens))
compressed_text_file.close()

ordered_tokens_frequency_file = open("digit_compressor/compressed/2-token-counts-order.txt", "r")
ordered_tokens = ordered_tokens_frequency_file.read().split('\n')
while '' in ordered_tokens: ordered_tokens.remove('')
# print(ordered_tokens)
ordered_tokens_frequency_file.close()

print('ordered_tokens is', ordered_tokens[:7])

def decompress_text():
    decompressed_text = ''
    for next_token in compressed_tokens:
        if (next_token.isdigit()):
            decompressed_text = decompressed_text + ordered_tokens[int(next_token)] + ' '
        else:
            decompressed_text = decompressed_text + next_token + ' '
    return decompressed_text.rstrip()

decompressed_text = decompress_text()

print(decompressed_text)

file = open('digit_compressor/compressed/3-decompressed-texts.txt', 'w')
file.write(decompressed_text)
file.close()

print("--- Decompression total execution Time: %s seconds ---" % round(time.time() - start_time, 1))