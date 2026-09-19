## What is my North Star? 

If I were to write a paper, then what do I want to show? 

1. DNA can be tokenized hierarchially  
2. This tokenizations scheme is biologically meaninfgul 
3. And performs better than other tokenization at some task(s) 


Right now: 
1. This point is given. DNA-HNET showed this. VQ-DNA as well
2. This point is not quite there. The current VQVAE encourages positional relationships
3. This is also not true

One thing I find interesting is that hierarchial tokenization is not mutually exclusive from other tokenization schemes. For example, one can do BPE and then tokenize hierarchially after. One could do KMERS and then tokenize hierarchially after. 


Problems with different tokenization schemes:

kmers
- a motif crossing a boundary is split across tokens
- the same motif is represented differenly depending on where the 4-mer boundaries fall

overapping k-mers
- fix the issue 
- have little sequence length compression

bpe 
- The same motif can be split differently depending on its surrounding sequence
- A single mutation can change several tokens and shift all subsequent tokens
- frequent sequences become tokens but rare ones may not


Potential issues with hierarchy 
- Instability of single code changes


I have an idea: That does not involve downsampling per se. 
Suppose we encode
Suppose we contextualize. Then we quantize. 

Anyway i dont know why I am looking at this as a possibility. It doesnt solve any particular problem. i was



# Tokenizer 
- To contextualize or not to contextualize: No contexualization 
- To group norm or not: No Group Norm: Layer Norm
- TO downsample or not: Downsample, convolution, Kernel 4, Stride 4
- Use fixed area pooling across all hierarchy scales
- Remove the refinement after preojection
- Use scales [1, 2, 4, 8, 16, 32, 64]
- Use codebook sizes k-128
- VQ loss: Encoder to final latent
- Encdr dim 128
- Latent dim 16 


# Coarse Codes as Long-Range Context

The tokenizer assigns one scale-1 code to each 256-base window, choosing from only eight codes. Because it encodes each non-overlapping 4-mer independently before averaging across the window, this code is a rough summary of learned 4-mer composition. Later scales encode corrections to that summary.

The scale-1 signal persists along DNA. In 9,856 pairs of adjacent windows, the code matched 49.8% of the time, compared with 13.1% for shuffled pairs. Across 616 regions of 8,192 bases, the most common code covered a median of 50% of the 32 windows. One *E. coli* region had an uninterrupted run of 11 windows, or 2,816 bases. The runs are not clean blocks, though: after runs of at least four windows, 74% of new codes lasted just one window, and 51% were followed immediately by a return to the old code. A region can favor one coarse code even when it has interruptions.

The resemblance largely stops at the coarse scale. When neighboring windows shared a scale-1 code, only 8.2% of corresponding scale-2 codes matched, versus 7.3% when scale 1 differed. Among changed scale-2 codes in matching scale-1 pairs, only 6.8% were nearest codebook neighbors, compared with 6.5% after shuffling. The coarse code therefore groups windows without specifying their finer details.

This pattern also appears across genomes, though the codes are not species labels. In four distant *E. coli* K-12 regions, codes 5 and 6 covered 80% of windows; in four *S. coelicolor* regions, code 0 covered 87%. *B. subtilis* also favored code 5.

## Finer Coarse Codes

The spatial pattern persists with 128 scale-1 codes. Adjacent 256-base windows chose the same code 5.2% of the time, compared with 0.8% for shuffled pairs. Across 616 regions of 8,192 bases, each region's 32 windows used a median of 22 distinct codes out of 128, compared with 4 out of 8 for the smaller codebook.

To test whether particular codes cluster within regions, the codes from all 616 regions were mixed and regrouped into sets of 32. The 16 most common codes within each real region covered 83.0% of its windows, compared with an average of 61.5% across shuffled groups, each using its own top 16. The real regions' top-16 codebook vectors had a mean pairwise distance of 0.54, versus 0.86 for the shuffled groups. For 35% of those codes, the nearest codebook neighbor was also in the region's top 16, versus 10% after shuffling. Even when neighboring windows chose different codes, their codebook vectors were about 40% closer than those of shuffled pairs. A larger codebook gives regions a richer subset of locally related coarse codes rather than making their choices spatially random.

This suggests a direction for NSM-DNA: model the sequence of coarse codes over long distances, then predict the finer scales. An 8,192-base region becomes just 32 scale-1 tokens, potentially enough context to learn when a compositional pattern persists, flickers, shifts, or returns. The prefix's own coarse-code history might be useful alongside, or instead of, its continuous latent representation.
