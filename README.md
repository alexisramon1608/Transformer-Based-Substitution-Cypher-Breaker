# Transformer-Based-Substitution-Cypher-Breaker
This is a repository for a school project of breaking a substitution based cypher.
The cypher replaces the top 256 most common character pairs with a symbol (as well as single unpaired characters with their own distinct symbols)
Notice that this makes it chaotic by nature, because a single character shift would change most of the encoded symbols
```
[hi][ m][y ][na][me][ i][s ][Al][ex]
[oh][ h][i ][my][ n][am][e ][is][ A][le][x]
(where top and bottom would be encoded by completely different symbols)
```
# Motivation
The motivation of using a transformer is that it can quickly learning token associations and encodings for sentence structures even if the encoded sample is very small (something brute force techniques struggle with). Think of it as having a smaller search space constrained by it's knowledge of language.

# Considerations
The input encodings have to be agnostic to the symbol dictionary we use (for example the symbol one could be associated with ab, or cb in two different encoded sequences)
To remedy that, I use an additional encoding on top of the symbol-encodings to only retain positional and frequency data