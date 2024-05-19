
# LLM based Classification (Supervised/Semi-Supervised)

## 1. Introduction

This repository is a mini-library (OOP style) to tackle NLP-based classification problems. It covers both semi-supervised and supervised learning based on (pre-trained) transformer models. In particular, we offer:

* **Data**: A NLP specific data-modeler transform DataFrame to DataLoader directly.
* **Model**: BERT + Customizable Classifier (*native*) or BertForSequenceClassification (*huggingface*).
* **Optimizer**: A BERT specific AdamW version allows: layer freezing, parameters regularization, lora fine-tuning.
* **Trainer**: A set of trainers offer state-of-art NLP classification algorithms, and easy to extend owing to OOP framework.

In addition, we provide two types of text data augmentation methods: 1) backtranslation; 2) prompt-based.

## 2. Library Architecture

<p align="center">
<img src="images\workflow.png" width="700">
Figure 1. Project Architecture/Workflow
</p>

## 3. Theory

### 3.1 Overview
To ease our presentation, let's introduce some main notations:

| Symbol     | Desc. |
| :--------   | :------- |
| $$L \equiv (\mathcal{X}, \mathcal{Y})$$   | Labeled examples, i.e., a sample pair $(x, y)$ is drawn from $\mathcal{S}$, where $x\in\mathcal{X}$, and $y\in\mathcal{Y}$|
| $$U \equiv \mathcal{X'}$$ | Unlabeled examples, i.e., a sample $x'$ is drawn from $\mathcal{X}'$ with potentially different distribution.    |
| $$\mathcal{L}, \mathcal{L}_{\texttt{sup}}, \mathcal{L}_{\texttt{unsup}}$$ | Loss function, supervised/unsupervised loss functions.|
| $p_1(y\|x)$ | Given an input $x$, the conditional class distribution predicted by model parameterized with $\theta$. |

In the case of supervised learning ($|U| = 0$), we usually require a large collection of labeled samples $L$ to train the NLP classifier. Unfortunately, in reality, labeled data are limited as the gathering process is labrious, time consuming and costly. Additionally, in a production environment, there could be two issues:
* **Noisy Data**: the distribution of "live" data can deviate from the training dataset.
* **Unexplored Data**: the amount of unlabeled data (live) is >>> labled data. Relying on humman annotation to utilize these unlabeled data is very inefficient.

*Semi-supervised learning* address this very issue. It leverages unlabeled data to mitigate the reliance on large labeled datasets. Sepcifically, besides the supervised loss (mostly cross-entropy), an unsupervised loss is introduced to track the "error" of unlabeled data. The ultimate loss function reads,
$$ \mathcal{L} = \mathcal{L}_{\texttt{sup}} + \lambda(t)\cdot\mathcal{L}_{\texttt{unsup}}$$
where $\lambda(t)$ is a ramp function that gradually increases the importance of unlabeled samples. The prevalent designs of unsupervised loss are via consistency regularization and psuedo labeling. In this project, we offer two main frameworks of this kind: 1) UDA; 2) MixMatch.

### 3.2 Consistency Training -- UDA/UDG

*Consistency regularization* ensures consistent prediction when noise and systematic perturbation is present. [Unsupervised Data Augmentation][1]</cite> (Xie et al, 2020) and [Towards Zero-Label Language Learning][2]</cite> (Wang et al, 2021) adopt this approach to tackle the unlabeled data. Essentially, for any unlabeled data, it generates an augmented version (discussed in *Section 4*), and requires the augmented version is classified the same as the original version.

<p align="left">
  <img src="images\uda.png"/>
  Figure 2. Taken from paper "Unsupervised Data Augmentation for Consistency Training". The right section showcases the mechanism to enforce consistent predictions between unlabeled data and augmenetation of them.
</p>

The author proposed couple of techniques to enhance the model performance, namely, *low confidence masking*, *sharpening prediction*. From my experiments, the low confidence masking seems to be the key. It mask out examples with low prediction confidence if lower than a threshold to prevent overfitting in the early stage of training, given very limited labeled data.

### 3.3 MixMatch ###

Before delving into MixMatch, let's first brief another important technique when dealing with unlabeled data -- *Pseudo Labeling*. It first predicts labels of unlabeled data and then trains the model on both labeled and unlabeled data simultaneously. The optimization is an iterative process as below:
- **Step 0**: set $\hat{L} = L$.
- **Step 1**: builds a classifier from currently labeled data $\hat{L}$.
- **Step 2**: use classifier from **Step 1** to predict labels for the unlabeled data $U$ and converts the most confident ones into labeled samples, i.e., $\hat{L} = \hat{L} \cup (\tilde{U},\,\texttt{Labeler}(\tilde{U}))$, where $\tilde{U}\subset U$.

[MixMatch][3]</cite> (Berthelot et al. 2019) combines both *consistency training* and *pseudo labeling*. There are $3$ components of the algorithm,
- **MixUp**: regularizes the model to favor simple linear behavior in-between training examples.
- **Consistency Training**: enforces the model to keep consistent predictions on perturbed unlabeled samples with unperturbed data.
- **Entropy Minimization**: train the model to make confident predictions on unlabeled data.

<p align="left">
  <img src="images\mixmatch.png"/>
  Figure 3. Taken from paper "MixMatch: A Holistic Approach to Semi-Supervised Learning".
</p>

The original MixMatch algorithm is not suitable for LLM-based classification, because BERT model needs both raw word embedding and attention masks, wihle it makes little sense to interpolate attention masks between examples. We instead defer the MixUp and blend two text samples at the level of BERT output (i.e., [CLS] token).

## 4. Data Augmentation

For all semi-supervised algorithm mentioned above, we need to perform data augmentation. In this project, we offer two types of data augmentation:

- **Back-Translation**: the procedure of translating an existing example $x$ in language $A$ into another language $B$ and then translating it back into $A$ to obtain an augmented example $\hat{x}$. The key of back-translation is to obtain diverse paraphrases while preserving the semantics of the original sentences.
- **Prompting**: the burst of GenAI offers us a powerful channel to generate text that is of similar semantics. As outlined by Wang et al in  [Towards Zero-Label Language Learning][2]</cite>, they use transformer-based models to generate unlabeled dataset with proper promoting design.

## Get Started

We provide handy classes and APIs for both analysis focused user and development focused user. There are $4$ jupyternote books to demonstrate data manipulation, data generation, optimization set-up, and transformer based classifiers. In addition, there are $3$ .py files to showcase end-to-end training of supervised/unsupervised model.


## Reference

- Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin A, *"MixMatch: A Holistic Approach to Semi-Supervised Learning"*, Advances in Neural Information Processing Systems (NIPS '19), vol 32, 2019, Curran Associates, Inc.

- Xie, Qizhe and Dai, Zihang and Hovy, Eduard and Luong, Minh-Thang and Le, Quoc V., *"Unsupervised data augmentation for consistency training"*, Advances in Neural Information Processing Systems (NIPS '20), vol 13, 2020, Curran Associates Inc.

- Zirui Wang and Adams Wei Yu and Orhan Firat and Yuan Cao, *"Towards Zero-Label Language Learning"*, ArXiv, 2021, abs/2109.


[1]: https://arxiv.org/pdf/1904.12848
[2]: https://arxiv.org/abs/2109.09193
[3]: https://proceedings.neurips.cc/paper_files/paper/2019/file/1cd138d0499a68f4bb72bee04bbec2d7-Paper.pdf







