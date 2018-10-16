#Modelling of air-quality in Belgium for forecasting purposes using Deep Neural Networks 

##CONTEXT
Forecasting of air pollutants, like the low-level (tropospheric) Ozone & Particulate Matter (PM), can accumulate and potentially harm natural ecosystems and human health.

##GOAL
To derive a Multiple-Input Multiple-Output (MIMO) Nonlinear Autoregressive Exogenous (NARX) model for the prediction of background Ozone and/or Particulate matter forecasting across Belgium, using a deep-learning [5] approach.
Evolutionary Algorithms (EA) and Reinforcement Learning (RL) techniques will be utilized, to tune hyper-parameters and discover new Neural Network Architectures [3, 4]. For example, Convolutional LSTM (Long Short Term Memory) models have been derived to capture spatio-temporal correlations [1]. Time-series cross-validation [2] techniques will assist the model selection.

##KIND OF WORK
Literature review and theoretical work 40% / Programming 40% / Writing 20%.

##PROFILE
Matlab or Python programming, machine learning theory/applications & data cleaning. Exploration of new multiple-output Deep Neural Network based approaches for spatial/time-series problems. Model selection and training will be mainly performed on a personally-owned workstation with a 6-core CPU, 64 GB Ram, and 2x CUDA capable GPUs.

###REFERENCES

[1] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, and Wang-chun Woo. 2015. Convolutional LSTM Network: a machine learning approach for precipitation nowcasting. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1 (NIPS'15), C. Cortes, D. D. Lee, M. Sugiyama, and R. Garnett (Eds.), Vol. 1. MIT Press, Cambridge, MA, USA, 802-810.

[2] Christoph Bergmeir, Rob J Hyndman, Bonsoo Koo (2018) A note on the validity of cross-validation for evaluating autoregressive time series prediction. Computational Statistics and Data Analysis, 120, 70-83.

[3] Neural Architecture Search with Reinforcement Learning, Barret Zoph, Quoc V. Le. International Conference on Learning Representations, 2017.

[4] Using Evolutionary AutoML to Discover Neural Network Architectures - https://ai.googleblog.com/2018/03/using-evolutionary-automl-to-discover.html

[5] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. 2016. Deep Learning. The MIT Press.

Ozone Narx DNN
Copyright (c) 2018, Konstantinos Theodorakos (email: madks@hotmail.com).
All rights reserved.

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.