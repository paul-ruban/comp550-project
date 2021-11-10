from collections import Counter
# from nltk.tokenize import word_tokenize
# import nltk

# Compressor

text_file = open("a-shade-of-halloween.epub.txt", "r")
text = text_file.read().replace('\n', ' ')
print(text)
# tokens = word_tokenize(text_file.read())
# print(tokens)
# print(len(tokens))
text_file.close()

# def count_tokens(tokens):
#     tokenCounts = Counter(tokens)
#     return tokenCounts

# tokenCounts = count_tokens(tokens)
# # tokenCountsOrder = list(tokenCounts.keys())

# def ordered_most_common_words(tokenCounts):
#     print(tokenCounts)
#     mostCommonWords = [word for word, wordCount in tokenCounts.most_common()]
#     return mostCommonWords

# tokenCountsOrder = ordered_most_common_words(tokenCounts)

# print(tokenCountsOrder[:5])

# def replaceTokensWithDigits(tokens, tokenCountsOrder):
#     compressedTokens = []
#     for token in tokens:
#         compressedTokens.append(tokenCountsOrder.index(token))
#     return compressedTokens

# compressedTokens = replaceTokensWithDigits(tokens, tokenCountsOrder)

# file = open('compressed-a-shade-of-halloween.epub.txt', 'w')
# for token in compressedTokens:
#     file.write(str(token) + ' ')
# file.close()

# print(tokenCountsOrder)

# file = open('token-counts-order.txt', 'w')
# for token in tokenCountsOrder:
#     file.write(token + "\n\n")
# file.close()
