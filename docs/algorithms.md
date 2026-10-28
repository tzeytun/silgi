# How silgi works

silgi uses classical image processing only: no neural networks, no pretrained
models, no network access. This page explains each method in plain language,
shows where it lives in the code and lists the papers it comes from.

Everything below shares one rule, enforced in a single place
([`core/roi.py`](../src/silgi/core/roi.py)): a method only ever sees the
selection's bounding box plus a margin, and its result is written back **only
where the mask is set**. Pixels outside the mask are never assigned, so they stay
bit-identical to the input. A test checks this for every method, every pixel
format and many random masks.

## Choosing a method

| Situation | Best method |
|---|---|
| You know the watermark (a PNG with transparency) | Reverse alpha |
| You have several images that share the same watermark | Estimate it, then reverse alpha |
| Thin text or lines on a smooth background | Fast marching (Telea) |
| A logo, stamp or object over texture | PatchMatch |
| A small blemish you can cover with a nearby patch | Heal / clone |

**Auto** ([`core/auto.py`](../src/silgi/core/auto.py)) follows the same table: a
known watermark means reverse alpha; a selection at most 6 px wide over a smooth
background (median local standard deviation of the ring around it below 0.02)
means Telea; everything else means PatchMatch. It always reports which method it
picked and why.

## Reverse alpha blending

*File: [`core/alpha.py`](../src/silgi/core/alpha.py), [`core/align.py`](../src/silgi/core/align.py)*

A semi-transparent watermark `W` with opacity `α` turns the original pixel `J`
into the observed pixel

```
I = α·W + (1 − α)·J
```

If `W` and `α` are known this can be solved exactly for `J`:

```
J = (I − α·W) / (1 − α)
```

That recovers the real picture underneath instead of inventing one. Two practical
limits:

* Where `α` is close to 1 the original is almost gone and rounding noise is
  amplified by `1 / (1 − α)`. Pixels at or above a threshold (default 0.85) are
  filled by inpainting instead.
* With 8-bit images the rounding error grows as `α` grows, so the guarantee
  "PSNR above 45 dB" holds for `α` up to about 0.5 at 8 bits and for higher
  opacity at 16 bits. The tests cover both.

**Finding the watermark.** Semi-transparent marks barely change colours but they
do create edges. silgi matches the gradient magnitude of the image against that
of the watermark's alpha with normalised cross-correlation, then refines the
position to sub-pixel precision by trying the neighbouring offsets and keeping
the one that leaves the least watermark edge energy after reversal.

### Estimating an unknown watermark

*File: [`core/alpha_estimate.py`](../src/silgi/core/alpha_estimate.py)*

When several images carry the same watermark (same size, similar place), silgi can
estimate it. This is a classical, simplified take on Dekel et al., *On the
Effectiveness of Visible Watermarks* (CVPR 2017), without any learning:

1. **Localise.** Photo content varies, the watermark does not. The per-pixel
   *median* of the image gradients keeps only the watermark's edges.
2. **Reconstruct and align.** Integrating the median gradients (a Poisson solve)
   gives the watermark's shape, which is used to align every image.
3. **Solve per pixel.** For each pixel `I_k = α·W + (1 − α)·J_k` across images
   `k`. The unknown backgrounds `J_k` are approximated by diffusing their
   surroundings into the watermark area; α and W then follow from least squares,
   first with a shared α, then with the colour fixed (most watermarks are one
   colour).
4. **Refine.** One global gain on α is tuned to leave the least edge energy.

Differences from the paper: no iterative multi-image matting optimisation and no
image priors. It needs roughly ten or more images with varied content. The
estimated watermark is shown to you for confirmation before it is used.

## PatchMatch

*File: [`core/patchmatch.py`](../src/silgi/core/patchmatch.py)*

Used for textured areas, the same family of algorithms as classic "content-aware
fill". The hole is filled with patches copied from the rest of the image.

1. **Pyramid.** Build a coarse-to-fine pyramid. At the coarsest level the hole is
   only a few patches wide and is initialised by diffusion.
2. **Search** (Barnes et al., *PatchMatch*, SIGGRAPH 2009). For every patch that
   overlaps the hole, find the most similar patch that is fully known. Start from
   random guesses, then repeat *propagation* (if my neighbour's best match is
   good, try the same offset for me) and *random search* (try random candidates
   at shrinking distances around the current best).
3. **Vote** (Wexler et al., *Space-Time Completion of Video*, TPAMI 2007). Every
   hole pixel becomes the weighted average of what the overlapping matched
   patches say it should be. Search and vote alternate several times per level.
4. **Upsample** the offsets and the image to the next level and repeat.

Presets: *Fast* (5×5 patches), *Balanced* (7×7), *Best* (9×9, more iterations).
You can restrict where patches may be copied from with the **source brush**.

**Determinism.** The search runs as red/black half-sweeps, so each pixel only
reads neighbours that are not being written in the same sweep, and every random
number comes from a counter-based hash of (seed, level, iteration, pixel). The
output is identical regardless of the number of threads, and the same input,
mask and seed always give the same bytes.

## Fast marching and Navier–Stokes

*File: [`core/diffusion.py`](../src/silgi/core/diffusion.py)*

Both are smooth fills that propagate colour from the edge of the hole inward.

* **Telea** (*An Image Inpainting Technique Based on the Fast Marching Method*,
  2004) fills pixels in order of their distance from the boundary, each as a
  weighted average of known neighbours that favours pixels close to and in the
  direction of the boundary.
* **Navier–Stokes** (Bertalmío, Bertozzi and Sapiro, 2001) treats image
  intensity like a fluid and continues the lines (isophotes) arriving at the hole.

They are very fast and ideal for thin text strokes on smooth backgrounds, but they
blur texture. silgi runs them per channel in float32, so 16-bit images and alpha
channels are handled like 8-bit RGB.

## Clone stamp and healing brush

*File: [`core/healing.py`](../src/silgi/core/healing.py)*

*Clone* copies pixels from a source area exactly. *Heal* copies only the source's
texture (its gradients) and solves a Poisson equation so brightness and colour
adapt to the surroundings. See the next section.

## Poisson blending and edge finishing

*Files: [`core/poisson.py`](../src/silgi/core/poisson.py), [`core/blend.py`](../src/silgi/core/blend.py)*

Pérez, Gangnet and Blake (*Poisson Image Editing*, SIGGRAPH 2003): inside the
region Ω find pixels `f` whose gradients match a guide image `g` while equal to
the original on Ω's boundary:

```
Δf = Δg  in Ω,      f = target  on ∂Ω
```

silgi uses its own sparse solver rather than `cv2.seamlessClone`, because that
function only accepts 8-bit three-channel images; this one handles 16-bit images
and alpha. Small regions use a direct sparse factorisation. Large regions start
from a coarse-grid solution (the correction `f − g` is smooth, so it survives
downsampling) followed by a limited number of conjugate-gradient iterations. This
made healing a 5 % area of a 12 MP image about 15× faster at a mean error of
about a tenth of a grey level.

Optional **edge finishing** works strictly inside the mask:

* *Feather* fades the fill into the original over a few pixels inward.
* *Poisson seam* re-solves the fill in the gradient domain so it meets its
  surroundings without a step in brightness or colour.

To blend over a wider band, grow the mask first. That is an explicit, undoable
selection change, so the guarantee about untouched pixels stays simple.

## Selection tools

*Package: [`core/select/`](../src/silgi/core/select/)*

* **Magic wand**: flood fill by per-channel colour tolerance, contiguous or global.
* **Color range**: every pixel within a ΔE distance in CIELAB of a picked colour.
* **Find text** (no OCR): glyph candidates come from MSER regions and from
  multi-scale white/black top-hat transforms (the latter find semi-transparent
  text that changes under the glyph). Each candidate must look like a character:
  plausible size and fill, roughly constant stroke width (measured with a
  distance transform). Glyphs are linked into lines only when they agree in
  height, position, spacing and stroke width; lines also need enough edge energy.
* **Find similar**: multi-scale normalised cross-correlation on gradient magnitude
  (robust to semi-transparent marks over changing backgrounds), greedy peak
  extraction and non-maximum suppression.
* **Grow, shrink, smooth, feather**: morphological operations on the mask.
  Feathering only ever goes inward.

## Colour, bit depth and metadata

* Pixels are processed as float32 in the image's *native* encoding. silgi does
  not convert to sRGB and back, because that would change pixels outside the
  mask. The ICC profile is embedded again on export and honoured only for the
  on-screen preview.
* 8- and 16-bit PNG and TIFF are read and written without loss; alpha channels
  are preserved; EXIF orientation is applied on load.
* EXIF is dropped by default (it can contain location and device details) and
  written back with the orientation tag reset when you ask for it.
* JPEG and lossy WebP cannot keep untouched pixels bit-identical. The export
  dialog says so.

## References

1. A. Telea. *An Image Inpainting Technique Based on the Fast Marching Method.*
   Journal of Graphics Tools, 9(1), 2004.
2. M. Bertalmío, A. Bertozzi, G. Sapiro. *Navier–Stokes, Fluid Dynamics, and
   Image and Video Inpainting.* CVPR 2001.
3. A. Criminisi, P. Pérez, K. Toyama. *Region Filling and Object Removal by
   Exemplar-Based Image Inpainting.* IEEE Transactions on Image Processing, 2004.
4. C. Barnes, E. Shechtman, A. Finkelstein, D. B. Goldman. *PatchMatch: A
   Randomized Correspondence Algorithm for Structural Image Editing.* ACM
   SIGGRAPH 2009.
5. Y. Wexler, E. Shechtman, M. Irani. *Space-Time Completion of Video.* IEEE
   TPAMI, 2007.
6. P. Pérez, M. Gangnet, A. Blake. *Poisson Image Editing.* ACM SIGGRAPH 2003.
7. T. Dekel, M. Rubinstein, C. Liu, W. T. Freeman. *On the Effectiveness of
   Visible Watermarks.* CVPR 2017.
