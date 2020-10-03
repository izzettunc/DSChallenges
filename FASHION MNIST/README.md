# Model

This network structure is inspired from googles' VGG NET and this is somewhat minified and modified version of it

![Network][network]


# Loss

Loss stops decreasing after 7th-ish epoch, which means I still need to work on that. I have tried to augment the dataset by mirroring images with different and multiple axes but no luck so far.

![Loss Plot][loss-plot]

[loss-plot]: plot.png
[network]: network.png
