# mlgreens
## ***M***achine ***L***earning for ***Green's*** functions
`mlgreens` is a package for predicting modal Green's functions from partial observations over domain spaces.

## Installation

Once the source code has been downloaded, you can install the package by running the following command:

```python3 -m pip install ./mlgreens```

from one directory above the top-level `mlgreens` folder containing *pyproject.toml*. If installation was successful, you should now be able to execute an import (`import mlgreens`) from a Python script or interactive editor.

## Documentation

Documentation for this package can be found on the wiki of this repository (coming soon).

## References

A number of machine learning approaches used here were adapted from previous work on the subject of transformer models for visual learning. This includes the transformer model itself and the positional encoding scheme first developed in Vaswani et al (2017):
* A. Vaswani, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017)
  * https://doi.org/10.48550/arXiv.1706.03762

The vision transformer (ViT) model used as the basis for the `MViTAE` autoencoder was proposed in Dosovitsky et al (2020):

* A. Dosovitskiy, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
  * https://doi.org/10.48550/arXiv.2010.11929

Finally, the use of ViT-style models for autoencoding masked visual information and much of the masking scheme used here was first developed in He et al (2022):

* K. He, et al. "Masked autoencoders are scalable vision learners." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2022)
  * https://doi.org/10.48550/arXiv.2111.06377
