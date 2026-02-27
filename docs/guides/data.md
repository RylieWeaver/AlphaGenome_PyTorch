# Data


## Metadata

It's recommended you read the `model.md` guide before this section, since much of the data's structure is a result of the architecture of the output heads.

The metadata object is a compact way to dictate and get relevant information for the output heads of the model. It is initialized with and contains an internal 'metadata' dictionary with the structure:
- `organisms`: ordered list of organism names
- `heads`: per-head config dict

For each enabled head, you define output size **per organism**:
- genome-track heads (`atac`, `dnase`, `procap`, `cage`, `rna_seq`, `chip_tf`, `chip_histone`), `contact_maps`, `splice_sites_classification`, and `splice_sites_usage` use `num_tracks: [T_org0, T_org1, ...]`
- splice-junction uses `num_tissues: [T_org0, T_org1, ...]`
- masked-language-modeling does not use tracks/tissues

For genome-track heads, metadata also stores `means` with shape `[O, max(T)]`. Those means are used inside the head to scale targets/predictions by division/multiplication, respectively. Because the `MultiOrganismLinear([O, D, T])` module is a dense weight tensor that always predicts `T` values, regardless of organism, some tracks/tissues are padded. These padded tracks/tissues are masked from the loss by passing in their respective tissue/track masks and should be given dummy mean values of 1.0. You can also call `metadata.make_all_masks()` to get the masks as a result of padding, but can also adjust it for any missing tracks/tissues for a specific data batch.

Note that there does NOT need to be a correspondence between the tracks/tissues of different organisms or task heads.  Track `i` in one organism (or one head) does not need to correspond to track `i` somewhere else.


## Batch
A `DataBatch` carries `dna_sequence`, `organism_index`, and whichever head targets/masks you are training on. Means are stored in the metadata, **not** the batch because they never vary by batch.

Shapes:
- `dna_sequence`: `[B, S, 4]`
- `organism_index`: `[B]`
- 1bp genome tracks (`atac/dnase/procap/cage/rna_seq`): data `[B, S, T]`, mask `[B, #S, T]`
- 128bp tracks (`chip_tf/chip_histone`): data `[B, S//128, T]`, mask `[B, #S, T]`
- contact maps: data `[B, S//2048, S//2048, T]`, mask `[B, #S//2048, #S//2048, T]`
- splice-site classification: `[B, S, C]`
- splice-site usage: data `[B, S, T]`, mask `[B, #S, T]`
- splice-site junction: data `[B, #D, #A, 2*T_tissues]` (doubled for both strands)
- MLM labels: `mlm [B, S]`

Masks Account for:
1. Padding tracks/tissues (same across dataset and can be gotten from metadata)
2. Missing-targets (varies by batch and must be incorporated by the user)

It is also up to the user to make sure that the targets are padded to `T` values for consistent shape. The masks are also helpful for this task with F.pad(), and any padded values should work as long as they are numerically valid (e.g. no NaNs/inf).

Most example scripts pull baseline masks from metadata, index by `organism_index`, and then pass them in batch.
