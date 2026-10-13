# Ideas, approaches and warnings

The user pastes approaches, ideas and warnings here. Agents read this file
before starting work, and annotate items they tested or know about. Never
edit or delete the user's text; add annotations underneath it.

Item format (number items so they can be cited):

```
## <n>. <short title>
<the user's text>

> **Annotation (YYYY-MM-DD, <agent / variant>):** tried / known / not yet tried;
> what happened, with bench numbers or a link to a report; status: works,
> does not work, mixed, untested.
```

Warnings apply to every agent until the user removes them.


## 1. Temperature or constraint strength conditioned training
What it sounds like. Varying temperature, or softening hard constraints, during training to allow for more experimentation, prevent mode collapse, etc. 

## 2. Truncated depth forward passes
Extremes are depth one forward passes from arbitrary levels, with a low-variance but biased gradient from the model recursively evaluated on child latents. 

## 3. Multi-pass sampling within hierarchy level
Get around issues with conditional independence assumptions within same color groups by simply sampling multiple times. May speed things up in net by allowing for fewer colors. In the limit, two color checkerboard would be possible. 

## 4. Masked diffusion sampling of channels (or factorized sampling)
Speed up the #channels sampling steps by switching from autoregressive on channels to masked diffusion within channels, or any other method which samples multiple channels simultaniously. 

---
