<div align="center">
<h1>
  <br>
  <a href="http://acids.ircam.fr"><img src="images/logo_acids.png" alt="ACIDS" width="200"></a>
  <br>
  Creative Machine Learning
  <br>
</h1>

<h3>Creative Machine Learning course and notebooks in JAX, PyTorch and Numpy.</h3>

<h4>
  <a href="#lessons">Lessons</a> •
  <a href="#setup">Setup</a> •
  <a href="#administrative">Administrative</a> •
  <a href="#details">Detailed lessons</a> •
  <a href="#contribution">Contribution</a> •
  <a href="#about">About</a>
</h4>
</div>

This repository contains the courses in machine learning applied to music and other creative mediums.
This course is currently given at the University of Tokyo (Japan), and along the [ATIAM Masters](http://atiam.ircam.fr) at IRCAM, Paris (France). 
The courses slides along with a set of interactive Jupyter Notebooks will be updated along the year to provide all the ML program.
This course is proudly provided by the <a href="http://acids.ircam.fr" target="_blank">ACIDS</a> team.
This course can be followed entirely online through the set of [Google slides](http://slides.google.com) and [Colab](http://colab.google.com) notebooks links provided openly along each lesson.
However, we do recommend to fork the entire environment and follow the interactive notebooks through Jupyter lab to develop your
own coding environment.

**As the development of this course is always ongoing, please pull this repo regularly to stay updated.**
**Also, please do not hesitate to post issues or PRs if you spot any mistake** (see the [contribution section](#contribution)).

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li> <a href="#lessons">Lessons</a>
      <ul>
        <li><a href="#lessons">Introduction</a></li>
        <li><a href="#lessons">Machine learning</a></li>
        <li><a href="#lessons">Neural networks</a></li>
        <li><a href="#lessons">Advanced networks</a></li>
        <li><a href="#lessons">Deep learning</a></li>
        <li><a href="#lessons">Bayesian inference</a></li>
        <li><a href="#lessons">Latent models</a></li>
        <li><a href="#lessons">Approximate inference</a></li>
        <li><a href="#lessons">Variational auto-encoders and flows</a></li>
        <li><a href="#lessons">Generative adversarial networks</a></li>
        <li><a href="#lessons">Diffusion models</a></li>
      </ul>
    </li>
    <li> <a href="#administrative">Datasets</a> </li>
    <li> <a href="#details">Detailed lessons</a> </li>
    <li> <a href="#contribution">Contribution</a> </li>
    <li> <a href="#about">About</a> </li>
  </ol>
</details>


## Lessons

---

**Quick explanation.** For each of the following lessons, you will find a set of badges containing links to different parts of the course, which allows you to follow either the _online_ or _offline_ version.
Simply click on the corresponding badge to follow the lesson.

|              Flavor | Type | Badge |
| -----------------: | --------------: | ---------: |
|            Online  |          Slides |            |
| ^^                 |       Notebook  |   3--5 ATP |
|            Offline |          Slides |      5 ATP |
| ^^                 |       Notebook  |            |

Note that if the badge is displayed in red color as follows 
[![Slides](https://img.shields.io/badge/Slides-none-7D1616.svg?style=flat-square&logo=googledrive)]() 
it means that the content is not available yet and will be uploaded later.


### [00 - Introduction](00_introduction.pdf)

[![Slides](https://img.shields.io/badge/Slides-online-7DA416.svg?style=flat-square&logo=googledrive)](https://github.com/acids-ircam/ddsp_pytorch) 
[![Powerpoint](https://img.shields.io/badge/Slides-download-167DA4.svg?style=flat-square&logo=files)](https://github.com/acids-ircam/ddsp_pytorch) 
[![Colab](https://img.shields.io/badge/Notebook-collab-7DA416.svg?style=flat-square&logo=googlecolab)](https://arxiv.org/abs/2001.04643) 
[![Notebook](https://img.shields.io/badge/Notebook-download-167DA4.svg?style=flat-square&logo=jupyter)](https://github.com/acids-ircam/ddsp_pytorch) 
[![Video on YouTube](https://img.shields.io/badge/Video-none-7D1616.svg?style=flat-square&logo=Youtube)]()
[![Published in ArXiV](https://img.shields.io/badge/Paper-none-7D1616.svg?style=flat-square&logo=arXiv)](https://arxiv.org/abs/2001.04643) 

This course provides a brief history of the development of artificial intelligence and introduces the general concepts of machine learning 
through a series of recent applications in the creative fields. This course also presents the pre-requisites, course specificities, toolboxes
and tutorials that will be covered and how to setup the overall environment.

---

### [01 - Machine learning](01_machine_learning.pdf)

- [Notebook - Machine learning](01a_machine_learning.ipynb)
- [Notebook - Feature-based learning](01b_feature_based_learning.ipynb)

[![Slides](https://img.shields.io/badge/Slides-none-7D1616.svg?style=flat-square&logo=googledrive)]() 
[![Powerpoint](https://img.shields.io/badge/Slides-none-7D1616.svg?style=flat-square&logo=files)]() 
[![Colab](https://img.shields.io/badge/Notebook-none-7D1616.svg?style=flat-square&logo=googlecolab)]() 
[![Notebook](https://img.shields.io/badge/Notebook-none-7D1616.svg?style=flat-square&logo=jupyter)]() 
[![Video on YouTube](https://img.shields.io/badge/Video-none-7D1616.svg?style=flat-square&logo=Youtube)]()
[![Published in ArXiV](https://img.shields.io/badge/Paper-none-7D1616.svg?style=flat-square&logo=arXiv)]() 
    
    
Problem statement
Classical problems
Linear models for regression and classification
Optimization and Initialization
Overfitting and cross-validation
Properties and complexity

---

## [02 - Neural networks](02_neural_networks.pdf)
    - [Notebook - Neural networks](02_neural_networks.ipynb)
Brief history
The artificial neuron
Geometric perspective on neurons
Gradient descent
Multi-layer perceptron
Backpropagation

- [03 - Advanced Neural Networks](03_advanced_networks.pdf)
    - [Notebook - Advanced networks](03_advanced_networks.ipynb)
Convolution
Convolutional NN
Recurrent NN
Regularization

- [04 - Deep learning](04_deep_learning.pdf)
    - [Notebook - Deep learning](04a_deep_learning.ipynb)
    - [Notebook - Auto-encoders](04b_auto_encoders.ipynb)
Residual networks
Attention and transformers
Auto-encoders
Modern applications

- [05 - Probabilities and Bayesian inference](04_probabilities_bayesian.pdf)
    - [Notebook - Probabilities and distributions](05a_probabilities.ipynb)
    - [Notebook - Bayesian inference](05b_bayesian_inference.ipynb)
Rules of probability
Conditional and marginal
Expectation
Notable distributions
Sampling
Bayesian probability
Probabilistic inference
Maximum A Posteriori
Maximum Likelihood
Conjugate distributions
    
- [06 - Latent models](06_latent_expectation_maximization.pdf)
    - [Notebook - Latent clustering and kMeans](06a_latent_models.ipynb)
    - [Notebook - Gaussian Mixture Models](06b_gaussian_mixture_models.ipynb)
Unsupervised learning
Clustering
Latent variables
Expectation-Maximization
Q-Function
Variational derivation
Gaussian Mixtures

- [07 - Approximate inference](07a_approximate_inference.ipynb)
    - [Notebook - Sampling](07b_sampling_mcmc.ipynb)
Sampling
Monte-Carlo and rejection
Metropolis-Hastings
Variational inference
Deriving variational inference
    
- [08 - Variational Auto-Encoder (VAE) and flows](08_variational_ae_flows.pdf)
    - [Notebook - Variational auto-encoders](08a_variational_auto_encoders.ipynb)
    - [Notebook - Normalizing flows](08b_normalizing_flows.ipynb)
Auto-Encoders
Variational Inference
VAEs and properties
beta-VAE and disentanglement
Normalizing flows
Flows in VAEs
    
- [09 - Adversarial learning](09_adversarial_learning.pdf)
    - [Notebook - Generative Adversarial Networks](09a_generative_adversarial_network.ipynb)
Estimating by comparing
Deriving the adversarial objective
Generative Adversarial Networks
Adversarial attacks
Flaws of GANs
Modern applications

- [10 - Diffusion models](10_diffusion_models.pdf)
    - [Notebook - Diffusion models](10a_diffusion_models.ipynb)
Diffusion models
Score-based approach
Langevin dynamics

- [11 - Guest lecture]

## Setup

Along the tutorials, we provide a reference code for each section. 
This code contains helper functions that will alleviate you from the burden of data import and other sideline implementations. 
You will find designated spaces in each file to develop your solutions. 
The code is in Python (notebooks impending) and relies on the concept of [code sections](https://fr.mathworks.com/help/matlab/matlab_prog/run-sections-of-programs.html),
 which allows you to evaluate only part of the code (to avoid running long import tasks multiple times and concentrate on the question at hand.

### Dependencies

#### Python installation

In order to get the baseline script to work, you need to have a working distribution of `Python 3.5` as a minimum (we also recommend to update your version to `Python 3.7`). We will also be using the following libraries

- [Matplotlib](https://matplotlib.org/)
- [Numpy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Scikit-Learn](https://scikit-learn.org/)
- [Music21](http://web.mit.edu/music21/)
- [Librosa](http://librosa.github.io/librosa/index.html)
- [PyTorch](https://pytorch.org/)

We highly recommend that you install [Pip](https://pypi.python.org/pypi/pip/) or [Anaconda](https://www.anaconda.com/download/) that will manage the automatic installation of those Python libraries (along with their dependencies). If you are using `Pip`, you can use the following commands

```
pip install matplotlib
pip install numpy
pip install scipy
pip install scikit-learn
pip install music21
pip install librosa
pip install torch torchvision
```

For those of you who have never coded in Python, here are a few interesting resources to get started.

- [TutorialPoint](https://www.tutorialspoint.com/python/)
- [Programiz](https://www.programiz.com/python-programming)

#### Jupyter notebooks and lab

In order to ease following the exercises along with the course, we will be relying on [**Jupyter Notebooks**](https://jupyter.org/). If you have never used a notebook before, we recommend that you look at their website to understand the concept. Here we also provide the instructions to install **Jupyter Lab** which is a more integrative version of notebooks. You can install it on your computer as follows (if you use `pip`)

```
pip install jupyterlab
```

Then, once installed, you can go to the folder where you cloned this repository, and type in

```
jupyter lab
```


## Administrative 

These administrative details concerns only the current physical session attached to the corresponding university.

### Course details

| Type | Information |
|----------:|:-------------|
| **Type**   | 2-credits graduate school course  |
| **Period** | April - July 2023 |
| **Span**   | 13 classes of 105 minutes |
| **Date**   | Thursday at 2:55 - 4:40 pm (JST Time) |
| **Onsite** | Room 214, 2nd Floor, Sci. 7 Building |
| **Online** | https://u-tokyo-ac-jp.zoom.us/j/81363691008?pwd=cmxiRURMdm9udXBKbTNjQkZvblNFQT09 |


### Full calendar

| Date | Course |
|----------:|:-------------|
| April 6 | 00 - Introduction |
| April 13 | 01 - Machine learning |
| April 20 | 02 - Neural networks |
| April 27 | 03 - Advanced networks |
| May 4 | **National holiday** |
| May 11 | 04 - Deep learning |
| May 18 | 05 - Bayesian inference |
| May 25 | 06 - Latent models |
| June 1 | **Midterm exam** |
| June 8 | 07 - Approximate inference |
| June 15 | 08 - VAEs and flows |
| June 22 | 09 - Adversarial networks |
| June 29 | 10 - Diffusion models |
| July 6 | 11 - Guest lecture #1 |
| July 13 | 12 - Guest lecture #2 |


## Contribution

Please take a look at our [contributing](CONTRIBUTING.md) guidelines if you're interested in helping!

## About

Code and documentation copyright 2012-2023 by all members of ACIDS. 

Code released under the [CC-BY-NC-SA 4.0 licence](https://creativecommons.org/licenses/by-nc-sa/4.0/).
