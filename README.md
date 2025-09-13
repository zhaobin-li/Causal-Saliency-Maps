# Causal Saliency Maps  

Generate causal saliency maps to explain CNN classifications using the Potential Outcomes Framework.  
This project unifies three XAI methods: [LIME](https://dl.acm.org/doi/10.1145/2939672.2939778), [RISE](http://bmvc2018.org/contents/papers/1064.pdf), and [Bayesian Teaching](https://www.nature.com/articles/s41598-021-89267-4).  

# Install

Required: PyTorch, Captum, scikit-image, scikit-learn, feather, tqdm, pytest. Optional: pytest-repeat, ipython

```
conda create --name causal
conda activate causal
conda install pytorch torchvision torchaudio cudatoolkit=11.3 captum scikit-image scikit-learn feather-format tqdm pytest pytest-repeat ipython -c pytorch -c conda-forge
```

Analysis: tidyverse, ggplot2, arrow. Optional: hmisc

```
conda create -n r_causal r-essentials r-base r-arrow r-hmisc -c conda-forge
```

# Run

```
cd /projects/f_ps848_1/zhaobin/causal
python -m pytest test
```

# Citation

Based on https://captum.ai/tutorials/Image_and_Text_Classification_LIME
