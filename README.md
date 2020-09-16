# my_segmentation

# 编译
cd awesome-semantic-segmentation-pytorch/core/nn

python setup.py build develop

| Backbone | Method |Basesize|Epoch|Segmodel | pixACC | mIOU |
|:-----:|:-----:|:-------:|:-----:|:-----:|:-----:|:-----:|
|R-18|Clear_Img+Rotation1.0| 480 | 60 |Deeplabv3|96.131|91.316|
|R-18|Clear_Img+Rotation0.5| 480 |60 |Deeplabv3|96.483|92.016|
|R-18|Clear_Img+Rotation0.0| 480 |60 |Deeplabv3|96.607|91.828|
|R-18|Clear_Img+Rotation0.5+RandomColor0.5| 480 |60 |Deeplabv3|96.139|92.335|
|R-18|Img+Rotation0.5+RandomColor0.5| 480 | 60 |Deeplabv3|97.371|93.330|
|R-18|Img+Rotation0.5+RandomColor0.5| 500 | 60 |Deeplabv3|96.861|92.224|
|R-18|Img+Rotation0.5+RandomColor0.5| 460 | 60 |Deeplabv3|97.735|92.883|
|R-18|Img+Rotation0.5+RandomColor0.5| 480 | 80 |Deeplabv3|96.851|92.668|
|R-18|Img+Rotation0.5+RandomColor0.5| 480 | 40 |Deeplabv3|97.221|92.580|
