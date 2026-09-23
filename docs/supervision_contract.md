# Supervision rules

SPHiVE-MT keeps temporal context and supervision separate.

1. **Frame validity**  
   `supervision: null` means the frame is context-only. It still updates temporal queries but contributes no matching or loss.

2. **Pixel validity**  
   `valid_mask=0` means unknown / ignore. These pixels are excluded from matching cost and mask losses.

3. **Taxonomy supervision**  
   Labels supervise the taxonomy node they are attached to. Coarser labels stay coarse; child classes are not guessed.

4. **Far vs. instance**  
   `far=false` gives instance supervision. `far=true` gives semantic supervision only. Both contribute to semantic masks.

5. **Negative information**  
   Only nodes listed in `exhaustive_node_ids` may be treated as fully known. Missing annotations outside those nodes remain unknown.

The loader may derive internal fields such as frame-valid masks, pixel-valid masks, and query-level negative masks from this metadata.
