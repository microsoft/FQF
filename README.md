# Fully parameterized Quantile Function (FQF)

Tensorflow implementation of paper

**[Fully Parameterized Quantile Function for Distribution Reinforcement Learning](https://arxiv.org/abs/1911.02140)**

Derek Yang, Li Zhao, Zichuan Lin, Tao Qin, Jiang Bian, Tie-yan Liu


If you use this code in your research, please cite
``` tex
@inproceedings{yang2019fully,
  title={Fully Parameterized Quantile Function for Distributional Reinforcement Learning},
  author={Yang, Derek and Zhao, Li and Lin, Zichuan and Qin, Tao and Bian, Jiang and Liu, Tie-Yan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6190--6199},
  year={2019}
}
```

## Requirements
- python==3.6
- tensorflow
- gym
- absl-py
- atari-py
- gin-config
- opencv-python

## Installation on Ubuntu
```bash
sudo apt-get update && sudo apt-get install cmake zlib1g-dev
pip install absl-py atari-py gin-config==0.1.4 gym opencv-python tensorflow-gpu==1.12.0
cd FQF
pip install -e .
```

## Experiments
- Our experiments and hyper-parameter searching can be simply run as the following
```bash
cd FQF/dopamine/discrete_domains
bash run-fqf.sh
```

## Bug Fixed
- It is recommended to use the L2 loss on gradient for probability proposal network, or clip the largest proposed probability to 0.98. The reason is as follows: in quantile function, when the probability goes to 1, the quantile value goes to infinity(or a very large number). Although a very large quantile value is reasonable for a probability such as 0.9999999, with limited approximation ability of neural network, quantile values for other probabilities will go up quickly, leading to a performance drop. 

## Acknowledgement
- Our code is implemented based on [dopamine](https://github.com/google/dopamine).


## Code of Conduct
- This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
