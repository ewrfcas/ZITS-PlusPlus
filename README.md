# ZITS++: Image Inpainting by Improving the Incremental Transformer on Structural Priors (TPAMI2023)

[Chenjie Cao*](https://ewrfcas.github.io/),
[Qiaole Dong*](https://dqiaole.github.io/),
[Yanwei Fu](http://yanweifu.github.io/)

[Project Page](https://ewrfcas.github.io/ZITS-PlusPlus/)

## Abstract

Image inpainting involves filling missing areas of a
corrupted image. Despite impressive results have been achieved
recently, restoring images with both vivid textures and reasonable
structures remains a significant challenge. Previous methods have
primarily addressed regular textures while disregarding holistic
structures due to the limited receptive fields of Convolutional
Neural Networks (CNNs). To this end, we study learning a Zeroinitialized
residual addition based Incremental Transformer on
Structural priors (ZITS++), an improved model upon our conference
work, ZITS (Dong et al. 2022). Specifically, given one corrupt
image, we present the Transformer Structure Restorer (TSR) module
to restore holistic structural priors at low image resolution,
which are further upsampled by Simple Structure Upsampler
(SSU) module to higher image resolution. To recover image texture
details, we use the Fourier CNN Texture Restoration (FTR) module,
which is strengthened by Fourier and large-kernel attention
convolutions. Furthermore, to enhance the FTR, the upsampled
structural priors from TSR are further processed by Structure
Feature Encoder (SFE) and optimized with the Zero-initialized
ResidualAddition (ZeroRA) incrementally. Besides, a newmasking
positional encoding is proposed to encode the large irregularmasks.
Compared with ZITS, ZITS++ improves the FTR’s stability and
inpainting ability with several techniques. More importantly, we
comprehensively explore the effects of various image priors for
inpainting and investigate how to utilize them to address high resolution
image inpainting with extensive experiments. This investigation
is orthogonal to most inpainting approaches and can
thus significantly benefit the community.


