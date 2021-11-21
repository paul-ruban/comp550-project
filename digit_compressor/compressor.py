from collections import Counter
from os import walk
import time

start_time = time.time()

fileNames = next(walk("digit_compressor/data"), (None, None, []))[2]  # [] if no file

print(fileNames)

# Compressor
def getAllTokens (fileNames):
    combinedTokens = []
    for fileName in fileNames:
        text_file = open("digit_compressor/data/" + fileName, "r")
        tokens = text_file.read().replace('\n', ' ').split(' ')
        while '' in tokens: tokens.remove('')
        # print(tokens[:10])
        # print(len(tokens))
        text_file.close()
        combinedTokens.extend(tokens)
        print("--- Tokenized File", fileName, "in %s seconds ---" % round(time.time() - start_time, 1))
    return combinedTokens

tokens = getAllTokens(fileNames)

# print(tokens)

def count_tokens(tokens):
    tokenCounts = Counter(tokens)
    return tokenCounts

tokenCounts = count_tokens(tokens)
# tokenCountsOrder = list(tokenCounts.keys())

print(tokenCounts)

def ordered_most_common_words(tokenCounts):
    # print(tokenCounts)
    mostCommonWords = [word for word, wordCount in tokenCounts.most_common()]
    return mostCommonWords

tokenCountsOrder = ordered_most_common_words(tokenCounts)

# print(tokenCountsOrder[:11])

def replaceTokensWithDigits(tokens, tokenCountsOrder):
    compressedTokens = []
    for token in tokens:
        compressedTokens.append(tokenCountsOrder.index(token))
    return compressedTokens

compressedTokens = replaceTokensWithDigits(tokens, tokenCountsOrder)
print(compressedTokens[:10])

file = open('digit_compressor/compressed/1-compressed-texts.txt', 'w')
for token in compressedTokens:
    file.write(str(token) + ' ')
file.close()

file = open('digit_compressor/compressed/2-token-counts-order.txt', 'w')
for token in tokenCountsOrder:
    file.write(token + "\n\n")
file.close()

print("--- Compression total execution Time: %s seconds ---" % round(time.time() - start_time, 1))