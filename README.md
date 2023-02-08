# The **BERT_Family** Package

Release (github): 2023.01.19.


## :earth_africa: Requirements
1. python 3.8
2. torch 1.13
3. transformers 4.25
---

## :triangular_flag_on_post:Overview

The **BERT_Family** package, **which aims to integrate all of the BERT variants to solve the NLP problem in real life**, implements the transformers module on [Hugging Face](https://huggingface.co/) to build a training pipeline in NLP downstream tasks.

### AutoModeling

As the following figure shows, the Automodeling can be split into three parts in the NLP task, **AutoDataset**, **Model Integration**, and **Downstream fine-tuning**. 

![](https://i.imgur.com/zUbON3V.png)


#### 1. AutoDataset: `set_dataset()`

Mostly, we spent lots of time in data preprocessing, such as cleaning, **tokenization**, packing into the dataset, etc. 

The **BERT_Family** provides `set_dataset()` funcion to solve these problems. The only thing user needs to do is load their data in `pandas.DataFrame` format. Then just feed the pd.DataFrame into `set_dataset()`, and everything is done!

#### 2. Model Integration: `load_model()`

Thanks for [Hugging Face](https://huggingface.co/), all of the BERT models can be found on the website. We use  `AutoTokenizer` and ` AutoModelForSequenceClassification` to implement the pre-trained tokenizer and pre-trained model.

So, everything about model details, usage, and the pre-trained algorithm is available on [Hugging Face](https://huggingface.co/). 

All you have to do is find a good pre-trained model [here](https://huggingface.co/models?sort=downloads&search=bert), and input it as a parameter `pretrained_model` or `tokenizer` in **BERT_Family**.

#### 3. Downstream Fine-tune: `train()`

When the model is initialized in your python environment, you can start to fine-tune your downstream task. 

The `train()` function is a basic training function in machine learning. You may use the training function yourself with any skills you want to add.



---
## :memo: Start from toy example

### Step 1: Install from github

```python=1
!pip install git+https://github.com/kiangkiangkiang/BERT_Family.git
```

The command `pip` or `pip3` depends on your environment. 


### Step 2: Initilize BERT_Family

```python=2
import BERT_Family as BF
myBF = BF.BFClassification(pretrained_model="albert_base_v2", tokenizer="albert_base_v2")
myBF.load_model()
```



> If `import BERT_Family` has "not found module" error, check the **target directory** where your package installed. Make sure this **target directory** is added in your environment. 
> An example from [colab](https://colab.research.google.com/drive/1NeurI_grWw_G-w8XosjY3elpdQrYE7U1?usp=sharing), see the block "Append path to environment (for Google Colab)" in the link. The **target directory** where you install  can be found from the output message of `!pip install ...`.




### Step 3: Set Data

```python=5
myBF.set_dataset(your_DataFrame, your_label)
```


### Step 4: Fine-Tune on your data!

```python=5
myBF.train(data=myBF.train_data_loader, epochs=10)
```

## :clipboard: More examples

- Google Colab: 
    1. https://colab.research.google.com/drive/1NeurI_grWw_G-w8XosjY3elpdQrYE7U1?usp=sharing
    2. https://colab.research.google.com/drive/15sQPgfSncybmcCp_1XydmHrmBugir_7y?usp=sharing

- GitHub (examples file): https://github.com/kiangkiangkiang/BERT_Family/tree/main/examples




## More details of language model (by chinese)!


| Title                     | Tutorials      |
| --------------------------|:-------------- |
| Language Model: BERT & GPT| [:link:][LM]   |
| BERT實戰篇 - Pytorch       | [:link:][BP]   |
| Summary of tokenization   | [:link:][TK]   |

[LM]: https://famous-grape-0b8.notion.site/Language-Model-BERT-GPT-a045fed5cabc4ddd9a725849ae4d6b8a
[BP]: https://famous-grape-0b8.notion.site/BERT-Pytorch-b5095cf86f654412af2557f461d8211c
[TK]: https://famous-grape-0b8.notion.site/f912141f14114bd2a85c3f6d31648705?v=8d50cbd977a3411d94fc36a76e0041fa

