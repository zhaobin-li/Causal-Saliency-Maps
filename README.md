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