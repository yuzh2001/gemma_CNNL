
## 移植过程

### 去除mmap
首先遇到的是

```sh
TypeError: Unpickler.__init__() got an unexpected keyword argument 'mmap'
```

参考 https://github.com/ggerganov/llama.cpp/issues/4319 解决了，注释掉了`mmap`。

```py

    def load_weights(self, model_path: str):
        self.load_state_dict(
            torch.load(
                model_path, 
                # mmap=True, 
                weights_only=True,
            )['model_state_dict'],
            strict=False,
        )
```
（已修改）

### 重新实现index_select

```sh
RuntimeError: "index_select_mlu" not implemented for 'ComplexFloat'
```

CNNL没有对complexFloat上的index_select实现。

```python
# model.py Gamma Forward

freqs_cis = []
for idx in input_positions:
    freqs_cis.append(self.freqs_cis[idx])
freqs_cis = torch.stack(freqs_cis).to('mlu')

# freqs_cis = self.freqs_cis.index_select(0, input_positions)
````

### 替换bfloat16

使用bfloat16的时候，会提示:
```sh
[2024-5-7 11:48:22] [CNNL] [Error]:[cnnlTransform] The bfloat16 data type supports only MLU590, 1V, or higher.
```

尝试把它转换为float，会提示：
```sh
[2024-5-7 11:56:2] [CNNL] [Error]:[cnnlCastDataType] CNNL_CAST_BFLOAT16_TO_FLOAT is supported only on MLU500 series.
```

因此需要在inference.py里手动配置一下dtype为float。

## 结果

### what is the name of the largest continent on earth?

```
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
```

### what is the difference between gemma 2b and gemma 2b-it model released by google
```
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
```

它还是会说胡话的，呃但是它毕竟是个2b模型，能要求多高呢~


