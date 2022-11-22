# RICE toolbox (Rotational Invariants of the Cumulant Expansion)
This '[MATLAB toolbox](https://github.com/NYU-DiffusionMRI/RICE)' contains all necessary functions for parameter estimation of the O(b^2) cumulant expansion for arbitrary axially symmetric b-tensors. Check [our recent paper](https://arxiv.org/pdf/2202.02399.pdf) for details on this implementation and [this book chaper](https://academic.oup.com/book/24921/chapter/188767614) for information on the cumulant expansion in general. Below we provide instructions on how to run the toolbox. See the '[example.m](https://github.com/NYU-DiffusionMRI/RICE)' script that performs the parameter estimation in an [example dataset](https://cai2r.net/resources/standard-model-of-diffusion-in-white-matter-the-smi-toolbox/).

The toolbox also allows the parameter estimation for minimal DKI and minimal RICE protocols.

<br>

## Overview: The cumulant expansion in diffusion MRI
<img width="1604" alt="Ob2_cumulant_expansion_RICE" src="https://user-images.githubusercontent.com/54751227/203182843-a1097e1d-9bd8-4a88-a60e-99f0d56a5104.png">


## LTE data
When only using LTE data, one can represent low-b data with the O(b) cumulant expansion as shown in Eq. (1). This is simply DTI, and it can represent DWIs up to ~b=1200 ms/mm^2.
For higher b-values (up to ~b=2500 ms/mm^2), one can represent DWIs with the O(b^2) cumulant expansion shown in Eq. (1). This is DKI.

## Multiple b-tensors
If we consider multiple b-tensor shapes as shown in the figure below, where β parametrizes the b-tensor shape. Most common examples are: β=1 for LTE (B has only one nonzero eigenvalue), β=0 for STE (B has 3 equal nonzero eigenvalues), and β=-1/2 for PTE (B has 2 equal nonzero eigenvalues).
<img width="1206" alt="axSymB" src="https://user-images.githubusercontent.com/54751227/197211877-1d589475-8835-4bcd-861a-35ee3f9a297f.png">
We see that for O(b) signals (Eq. (3) ), the representation does not change (still DTI).
However, for O(b^2) a new tensor shows up: the diffusion covariance tensor, C, see Eq. (4). C is more general than kurtosis, actually it contains all the information of the kurtosis tensor plus some extra.

## RICE Authors
- [Santiago Coelho](https://santiagocoelho.github.io/)
- [Els Fieremans](https://www.diffusion-mri.com/who-we-are/els-fieremans/)
- [Dmitry Novikov](https://www.diffusion-mri.com/who-we-are/dmitry-novikov/)

Do not hesitate to reach out to Santiago.Coelho@nyulangone.org (or [@santicoelho](https://twitter.com/santicoelho) in Twitter) for feedback, suggestions, questions, or comments[^note].

## LICENSE

A [US patent](link to patent) contains some of the related developments. 

```
%  Authors: Santiago Coelho (santiago.coelho@nyulangone.org), Els Fieremans, Dmitry Novikov
%  Copyright (c) 2022 New York University
%              
%   Permission is hereby granted, free of charge, to any non-commercial entity ('Recipient') obtaining a 
%   copy of this software and associated documentation files (the 'Software'), to the Software solely for
%   non-commercial research, including the rights to use, copy and modify the Software, subject to the 
%   following conditions: 
% 
%     1. The above copyright notice and this permission notice shall be included by Recipient in all copies
%     or substantial portions of the Software. 
% 
%     2. THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
%     NOT LIMITED TO THE WARRANTIESOF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
%     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BELIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
%     WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF ORIN CONNECTION WITH THE
%     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
% 
%     3. In no event shall NYU be liable for direct, indirect, special, incidental or consequential damages
%     in connection with the Software. Recipient will defend, indemnify and hold NYU harmless from any 
%     claims or liability resulting from the use of the Software by recipient. 
% 
%     4. Neither anything contained herein nor the delivery of the Software to recipient shall be deemed to
%     grant the Recipient any right or licenses under any patents or patent application owned by NYU. 
% 
%     5. The Software may only be used for non-commercial research and may not be used for clinical care. 
% 
%     6. Any publication by Recipient of research involving the Software shall cite the references listed
%     below.
%
% REFERENCES:
% - Coelho, S., Fieremans, E., Novikov, D.S., 2022. RICE Arxiv info.
```


[^note]:
    Please cite these works if you use the SMI toolbox in your publication:
    - Coelho, S., Fieremans, E., Novikov, D.S., 2022. RICE Arxiv info
