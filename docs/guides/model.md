# Model


## Base Model
AlphaGenome follows an **[Encoder -> Transformer -> Decoder]** architecture.

The encoder is a stack of convolutions and max pools that coarsens the sequence from **1 bp** resolution to **128 bp** resolution. The transformer tower then operates over those 128-bp tokens, with it's attention bias modulated by a Row-Attention module that performs pair-to-pair attention at an even coarser **(2048 x 2048) bp** resolution (scales as O(S^3) where S is the sequence length, but at an extremely coarse resolution). Finally, the decoder upsamples back to base-pair resolution with U-net-style skip connections from the higher-resolution encoder features.

![AlphaGenome Model Architecture and Row Attention](../../images/AG_Total.png)

The base model outputs embeddings at:
- **1 bp** sequence resolution
- **128 bp** sequence resolution
- **(2048 x 2048) bp** pair resolution

Colloquially, the base model of AlphaGenome operates as a U-net with Transformer blocks in the middle, with the caveat that the Transformer blocks are modulated by a pair-to-pair attention bias.


## Output Heads
Most output heads use a `MultiOrganismLinear()` layer, which stores a dense weight tensor of shape **[O, D, T]**:
- `O`: number of organisms
- `D`: embedding dimension
- `T`: max output size (tracks/tissues) across organisms for that head

At runtime, each sample is routed through the organism-specific slice of that tensor.

Example: with 2 organisms (human/mouse), `D=768`, and RNA-seq tracks `[200, 150]`, the RNA-seq projection weight is `[2, 768, 200]`. Human uses one `[768, 200]` matrix, mouse uses the other.

Using multiple tracks for the same output head allows the AlphaGenome model to account for external factors that are not solely a function of DNA. For example, two samples with identical DNA sequences may have different RNA-Seq gene expression because they were extracted from different tissues or exposed to different environmental conditions. In this case, each [tissue x environment] pair can be given a different track.

Exceptions: 
- The **Masked Language Modeling** head uses a standard `Linear()` projection from 1-bp embeddings to vocab logits.
- The **Splice-junction** head uses **tissues** (not tracks) and always predicts both strands, so output channels are **`2*T`**.
