ytorch(pytorch) root@notebook-devenviron-0507-110018-cf0l5r-notebook-0:/workspace/algorithm/transfer/zhineng_exp# python inference.py 
/workspace/algorithm/transfer/zhineng_exp/gemma/gemma/model.py:530: UserWarning:  MLU operators don't support 64-bit calculation. so the 64 bit data will be forcibly converted to 32-bit for calculation.  (Triggered internally at /torch/catch/torch_mlu/csrc/aten/utils/tensor_util.cpp:159.)
  token_ids_tensor = token_ids_tensor.to(device)
[2024-5-7 14:52:45] [CNNL] [Warning]:[cnnlRandCreateGenerator_v2] will be deprecated.
sunami Warning System Project on python sklearn,and keras
sunami Warning System Project on python sklearn,and keras
sunami Warning System Project on python sklearn,and keras
sunami Warning System Project on python sklearn,and keras
sunami Warning System Project on python sklearn,and keras
sunami Warning System Project on python sklearn,and keras


Answer:  

Step 1/4
1. The largest continent in the world is called the "blue continent" by some people. This name is not very popular, as this continent is often referred to as Asia, Europe, Africa, South America, or North America.

Step 2/4
2. However, the earth has seven continents, each with its geographical features, climate, languages, and cultures. The largest continent is actually not the largest by area, but by population.

Step 3/4
3. The earth's largest continent is indeed Africa, which is home to over 1 billion people, more than twice the population of the next largest continent, Asia. Africa also has vast expanses of wilderness and natural resources, including oil, natural gas, and raw materials.

Step 4/4
4. Therefore, the largest continent on earth is actually Africa, not some hypothetical "blue continent" or any other specific continent.


 in April 2016? i mean what is new in this version?
(1) this model have more parameters in general weights file and data, for example: fc5, block_scale_factor, r50, r101, n224, n264, n384, n408, n544
(2) what is new in this model i am wondering. because fc5 is used to replace the last two dense layers (fc6,fc7) of resnet-101. I think i do not want this model.
thank you very much for your prompt reply!
it will be great if you can give a link to download the model weights
@haozou is there a download link of these weights?
There are not pretrained weights for this model yet. You should train the weights for this model by yourself.
https://howtodothings.com/cs/object-detection
The dataset is publicly available in the <code>models/research</code> directory, which ships with TensorFlow. Open Computer Vision Projects
Actually, with VGG and VGGNet (VGG-16,VGG-19,VGG-22), the pretrained model means you just load the weights, and use it as if a pre-trained model, no need to train.