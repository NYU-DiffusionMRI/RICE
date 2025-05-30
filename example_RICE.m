%% Example code for RICE (Rotational Invariants of the Cumulant Expansion)
% parameter estimation with the RICE toolbox
%
% EXAMPLE 1 - Fitting full RICE on LTE+PTE data. Plotting all RICE maps and
%             DKI maps + microscopic FA + size-shape covariance (SSC)
%
% EXAMPLE 2 - Fitting minimal DKI vs full DKI on LTE data.
%
% EXAMPLE 3 - Fitting minimal RICE (LTE + STE) vs full RICE (LTE + PTE)
%
%
% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
%% EXAMPLE 1 - Fitting full RICE on LTE+PTE data. Plotting all RICE maps and
clc,clear,close all

% Path were the test dataset is located
pathFiles = '/Users/coelhs01/Documents/SantiagoCoelho/Git/RICE_example_datasets/dataset_1';

% Load example dataset 1, mask, and protocol
nii_dwi = load_untouch_nii(fullfile(pathFiles,'dwi.nii'));
DWI = nii_dwi.img;

b = load(fullfile(pathFiles,'dwi.bval'));
beta = load(fullfile(pathFiles,'dwi.beta'));
dirs = load(fullfile(pathFiles,'dwi.dirs'));

nii_mask = load_untouch_nii(fullfile(pathFiles,'mask.nii'));
mask = logical(nii_mask.img);

% RICE toolbox parameter estimation example
type = 'fullRICE';  %  Estimate full D and C tensors from LTE + PTE data (WLLS)
CSphase = 1;        % Use Condon-Shortley phase in spherical harmonics definition
ComplexSTF = 0;     % Use real-valued spherical harmonics definition
nls_flag = 1;       % Use local nonlinear smoothing for fitting to boost SNR
parallel_flag = 1;  % Use paralellization
[b0, tensor_elems, RICE_maps, DIFF_maps] = RICEtools.fit(DWI, b, dirs, beta, mask, CSphase, ComplexSTF, type, nls_flag, parallel_flag);
% tensor_elems contains Slm and Alm elements

% Compute fiber basis projections (axial and radial diffusivities and kurtosis)
DKI_maps = RICEtools.get_DKI_fiberBasis_maps_from_4D_DW_tensors(tensor_elems, mask, CSphase, ComplexSTF);

% Plot color FA map
slice = 40;
unitBrightness = repmat(DKI_maps.fa(:,:,slice),1,1,3);
MainEigvec_scaled = uint8(abs(squeeze(DKI_maps.fe(:,:,slice,:))).*unitBrightness*255);
figure('Position',[527 467 1033 871]), image(flip(permute(MainEigvec_scaled,[2 1 3]),1)), axis off equal

% Plot RICE maps
RICE_maps.W0 = 3*RICE_maps.S0./(RICE_maps.D0).^2;
RICE_maps.W2 = 3*RICE_maps.S2./(RICE_maps.D0).^2;
RICE_maps.W4 = 3*RICE_maps.S4./(RICE_maps.D0).^2;
names = {'$D_0$','$D_2$','$W_0$','$W_2$','$W_4$','$A_0$','$A_2$'};
clims = [0 3;0 1.5;0 2;0 1.5;0 0.5;-0.5 0.5;0 1];
allRICE = cat(4,RICE_maps.D0,RICE_maps.D2,RICE_maps.W0,RICE_maps.W2,RICE_maps.W4,RICE_maps.A0,RICE_maps.A2);
figure('Position',[201 232 1971 910]), colormap gray
RICEtools.WrapperPlotManySlices(permute(allRICE,[2 1 3 4]), slice, clims, names, 2, [], 1, 1),

% Plot DKI maps + microscopic FA + size-shape covariance (SSC)
allDIFF = cat(4,RICE_maps.D0,DKI_maps.fa,DKI_maps.mw,DKI_maps.aw,DKI_maps.rw,DIFF_maps.ufa,DIFF_maps.SSC);
clims = [0 3;0 1;0 2;0 1.5;0 3;0 1;0 1];
names = {'MD','FA','MK','AK','RK','$\mu$FA','SSC'};
figure('Position',[201 232 1971 910]), colormap gray
RICEtools.WrapperPlotManySlices(permute(allDIFF,[2 1 3 4]), slice, clims, names, 2, [], 1, 1),


% Computing SA and TQ decompositions
Dlm = tensor_elems(:,:,:,1:6);
Slm = tensor_elems(:,:,:,7:21); % tensor_elems contains Slm and Alm elements of C
Alm = tensor_elems(:,:,:,22:27); % tensor_elems contains Slm and Alm elements of C

% Compute TQ decomposition
Q00 = 5/9 * Slm(:,:,:,1) + 2/9 * Alm(:,:,:,1) ;
T00 = 4/9 * Slm(:,:,:,1) - 2/9 * Alm(:,:,:,1) ;
Q2m = 7/9 * Slm(:,:,:,2:6) - 2/9 * Alm(:,:,:,2:6) ;
T2m = 2/9 * Slm(:,:,:,2:6) + 2/9 * Alm(:,:,:,2:6) ;
T4m = Slm(:,:,:,7:15);
Tlm = cat(4,T00,T2m,T4m);
Qlm = cat(4,Q00,Q2m);

%% plot all rotational invariants of D
clc,close all
mask_RICE = logical(mask); mask_RICE(:,:,[1:slice-1 slice+1:size(mask,3)]) = 0;
tic
D_RICE = RICEtools.ComputeInvariantsFromCumulants(Dlm,'D',mask_RICE,CSphase,ComplexSTF);
invariants_D = cat(4,D_RICE.D_0, sqrt(D_RICE.D_22), nthroot(D_RICE.D_23,3));
nametags={'$\mathsf{D}_{0}$','$\mathsf{D}_2 = \mathsf{D}_{2|2}$','$\mathsf{D}_{2|3}$'};
clims=[0 3;0 1;-0.5 1];
figure('Position', [519 421 1657 628])
RICEtools.WrapperPlotManySlices(permute(invariants_D,[2 1 3 4]), slice,clims,nametags,1,[],1),

%% plot all intrinsic rotational invariants of S and A
clc,close all
C = cat( 4, Slm, Alm);
mask_RICE = logical(mask); mask_RICE(:,:,[1:slice-1 slice+1:size(mask,3)]) = 0;
C_RICE = RICEtools.ComputeInvariantsFromCumulants(C,'C',mask_RICE,CSphase,ComplexSTF);

intrinsic_C = cat(4,C_RICE.S_0, sqrt(C_RICE.S_22), nthroot(C_RICE.S_23,3), ...
                     sqrt(C_RICE.S_42), nthroot(C_RICE.S_43,3), nthroot(C_RICE.S_44,4), nthroot(C_RICE.S_45,5), ...
                     nthroot(C_RICE.S_E,3), nthroot(C_RICE.S_Et,3),...
                     C_RICE.A_0, sqrt(C_RICE.A_22), nthroot(C_RICE.A_23,3));
nametags={'$\mathsf{S}_{0}$','$\mathsf{S}_2 = \mathsf{S}_{2|2}$','$\mathsf{S}_{2|3}$',...
          '$\mathsf{S}_4 = \mathsf{S}_{4|2}$','$\mathsf{S}_{4|3}$','$\mathsf{S}_{4|4}$','$\mathsf{S}_{4|5}$','$\mathsf{S}_{4|6}$','$\mathsf{S}_{4|7}$',...
          '$\mathsf{A}_{0}$','$\mathsf{A}_2 = \mathsf{A}_{2|2}$','$\mathsf{A}_{2|3}$'};
clims = [ 0 0.5;0 0.5;-0.3 0.3;0 0.5;0 0.5;-0.3 0.3; ...
          0 0.5;-2 -1;-0.3 0.3;0 2;0 1;-0.3 0.3 ];
figure('Position', [22 192 2501 927])
RICEtools.WrapperPlotManySlices(permute(intrinsic_C,[2 1 3 4]), slice,clims,nametags,2,[],1),

mixed_C = cat(4,C_RICE.RS2S4_phi,C_RICE.RS2S4_theta,C_RICE.RS2S4_psi,C_RICE.RS2A2_phi,C_RICE.RS2A2_theta,C_RICE.RS2A2_psi);
nametags={'$\alpha_{\mathsf{S}^{(4)}}$','$\beta_{\mathsf{S}^{(4)}}$','$\gamma_{\mathsf{S}^{(4)}}$',...
          '$\alpha_{\mathsf{A}^{(2)}}$','$\beta_{\mathsf{A}^{(2)}}$','$\gamma_{\mathsf{A}^{(2)}}$'};
clims = [ 0 pi ; -pi pi ; 0 pi ; 0 pi ; -pi pi ; 0 pi ];
figure('Position', [22 192 2501 927])
RICEtools.WrapperPlotManySlices(permute(mixed_C,[2 1 3 4]), slice,clims,nametags,1,[],1),

%% plot all intrinsic and mixed rotational invariants of T and Q
clc,close all
C = cat( 4, Tlm, Qlm);
mask_RICE = logical(mask); mask_RICE(:,:,[1:slice-1 slice+1:size(mask,3)]) = 0;
C_RICE = RICEtools.ComputeInvariantsFromCumulants(C,'C',mask_RICE,CSphase,ComplexSTF);

intrinsic_C = cat(4,C_RICE.S_0, sqrt(C_RICE.S_22), nthroot(C_RICE.S_23,3), ...
                     sqrt(C_RICE.S_42), nthroot(C_RICE.S_43,3), nthroot(C_RICE.S_44,4), nthroot(C_RICE.S_45,5), ...
                     nthroot(C_RICE.S_E,3), nthroot(C_RICE.S_Et,3),...
                     C_RICE.A_0, sqrt(C_RICE.A_22), nthroot(C_RICE.A_23,3));
nametags={'$\mathsf{T}_{0}$','$\mathsf{T}_2 = \mathsf{T}_{2|2}$','$\mathsf{T}_{2|3}$',...
          '$\mathsf{T}_4 = \mathsf{T}_{4|2}$','$\mathsf{T}_{4|3}$','$\mathsf{T}_{4|4}$','$\mathsf{T}_{4|5}$','$\mathsf{T}_{4|6}$','$\mathsf{T}_{4|7}$',...
          '$\mathsf{Q}_{0}$','$\mathsf{Q}_2 = \mathsf{Q}_{2|2}$','$\mathsf{Q}_{2|3}$'};
clims = [ 0 0.5;0 0.5;-0.3 0.3;0 0.5;0 0.5;-0.3 0.3; ...
          0 0.5;-2 -1;-0.3 0.3;0 2;0 0.5;-0.3 0.3 ];
figure('Position', [22 192 2501 927])
RICEtools.WrapperPlotManySlices(permute(intrinsic_C,[2 1 3 4]), slice,clims,nametags,2,[],1),

mixed_C = cat(4,C_RICE.RS2S4_phi,C_RICE.RS2S4_theta,C_RICE.RS2S4_psi,C_RICE.RS2A2_phi,C_RICE.RS2A2_theta,C_RICE.RS2A2_psi);
nametags={'$\alpha_{\mathsf{T}^{(4)}}$','$\beta_{\mathsf{T}^{(4)}}$','$\gamma_{\mathsf{T}^{(4)}}$',...
          '$\alpha_{\mathsf{Q}^{(2)}}$','$\beta_{\mathsf{Q}^{(2)}}$','$\gamma_{\mathsf{Q}^{(2)}}$'};
clims = [ 0 pi ; -pi pi ; 0 pi ; 0 pi ; -pi pi ; 0 pi ];
figure('Position', [22 192 2501 927])
RICEtools.WrapperPlotManySlices(permute(mixed_C,[2 1 3 4]), slice,clims,nametags,1,[],1),


% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
% =========================================================================================================
%% EXAMPLE 2 - Fitting minimal DKI vs full DKI on LTE data.
clc,clear,close all

% Path were the test dataset is located
pathFiles = '/Users/coelhs01/Documents/SantiagoCoelho/Git/RICE_example_datasets/dataset_2';

% Load example dataset 2, mask, and protocol
nii_dwi = load_untouch_nii(fullfile(pathFiles,'dwi_minimalDKI.nii'));
DWI_minimalDKI = nii_dwi.img;
b_minimalDKI = load(fullfile(pathFiles,'dwi_minimalDKI.bval'));
beta_minimalDKI = load(fullfile(pathFiles,'dwi_minimalDKI.beta'));
dirs_minimalDKI = load(fullfile(pathFiles,'dwi_minimalDKI.dirs'));

nii_dwi = load_untouch_nii(fullfile(pathFiles,'dwi_fullDKI.nii'));
DWI_fullDKI = nii_dwi.img;
b_fullDKI = load(fullfile(pathFiles,'dwi_fullDKI.bval'));
beta_fullDKI = load(fullfile(pathFiles,'dwi_fullDKI.beta'));
dirs_fullDKI = load(fullfile(pathFiles,'dwi_fullDKI.dirs'));

nii_mask = load_untouch_nii(fullfile(pathFiles,'mask.nii'));
mask = logical(nii_mask.img);

CSphase = 1;        % Use Condon-Shortley phase in spherical harmonics definition
ComplexSTF = 0;     % Use complex-valued spherical harmonics definition
nls_flag = 1;       % Use local nonlinear smoothing for fitting to boost SNR
parallel_flag = 1;  % Use paralellization

type = 'minimalDKI';  % Estimate minimal DKI, Eq. (30) in RICE paper (WLLS)
[b0_a, tensor_elems_a, RICE_maps_a, DIFF_maps_a] = RICEtools.fit(DWI_minimalDKI, b_minimalDKI, dirs_minimalDKI, beta_minimalDKI, mask, CSphase, ComplexSTF, type, nls_flag, parallel_flag);

type = 'fullDKI';  % Estimate DKI (WLLS)
[b0_b, tensor_elems_b, RICE_maps_b, DIFF_maps_b] = RICEtools.fit(DWI_fullDKI, b_fullDKI, dirs_fullDKI, beta_fullDKI, mask, CSphase, ComplexSTF, type, nls_flag, parallel_flag);

% Plot comparison for MD, FA, MK maps
slice = 40;
allMAPS = cat(4,DIFF_maps_a.md,DIFF_maps_a.fa,DIFF_maps_a.mw,DIFF_maps_b.md,DIFF_maps_b.fa,DIFF_maps_b.mw);
allMAPS = permute(allMAPS,[2 1 3 4]);
clims = [0 3;0 1;0 2;0 3;0 1;0 2];
nametags = {'MD minimal DKI','FA minimal DKI','MK minimal DKI','MD DKI','FA DKI','MK DKI'};
figure('Position', [293 304 1683 908]), colormap gray
RICEtools.WrapperPlotManySlices(allMAPS, slice,clims,nametags,2,[],1),

%% EXAMPLE 3 - Fitting minimal RICE vs full RICE on LTE+STE data.
clc,clear,close all

% Path were the test dataset is located
pathFiles = '/Users/coelhs01/Documents/SantiagoCoelho/Git/RICE_example_datasets/dataset_3';

% Load example dataset 3, mask, and protocol
nii_dwi = load_untouch_nii(fullfile(pathFiles,'dwi_minimalRICE.nii'));
DWI_minimalRICE = nii_dwi.img;
b_minimalRICE = load(fullfile(pathFiles,'dwi_minimalRICE.bval'));
beta_minimalRICE = load(fullfile(pathFiles,'dwi_minimalRICE.beta'));
dirs_minimalRICE = load(fullfile(pathFiles,'dwi_minimalRICE.dirs'));

nii_dwi = load_untouch_nii(fullfile(pathFiles,'dwi_fullRICE.nii'));
DWI_fullRICE = nii_dwi.img;
b_fullRICE = load(fullfile(pathFiles,'dwi_fullRICE.bval'));
beta_fullRICE = load(fullfile(pathFiles,'dwi_fullRICE.beta'));
dirs_fullRICE = load(fullfile(pathFiles,'dwi_fullRICE.dirs'));

nii_mask = load_untouch_nii(fullfile(pathFiles,'mask.nii'));
mask = logical(nii_mask.img);

CSphase = 1;        % Use Condon-Shortley phase in spherical harmonics definition
ComplexSTF = 0;     % Use complex-valued spherical harmonics definition
nls_flag = 1;       % Use local nonlinear smoothing for fitting to boost SNR
parallel_flag = 1;  % Use paralellization

type = 'minimalRICE'; % Estimate minimal RICE, Eq. (31) in RICE paper (WLLS)
[b0_a, tensor_elems_a, RICE_maps_a, DIFF_maps_a] = RICEtools.fit(DWI_minimalRICE, b_minimalRICE, dirs_minimalRICE, beta_minimalRICE, mask, CSphase, ComplexSTF, type, nls_flag, parallel_flag);

type = 'fullRICE';    % Estimate RICE (WLLS)
[b0_b, tensor_elems_b, RICE_maps_b, DIFF_maps_b] = RICEtools.fit(DWI_fullRICE, b_fullRICE, dirs_fullRICE, beta_fullRICE, mask, CSphase, ComplexSTF, type, nls_flag, parallel_flag);

% Plot comparison for MD, FA, MK maps
slice = 40;
allMAPS = cat(4,DIFF_maps_a.md,DIFF_maps_a.fa,DIFF_maps_a.mw,DIFF_maps_a.ufa,DIFF_maps_b.md,DIFF_maps_b.fa,DIFF_maps_b.mw,DIFF_maps_b.ufa);
allMAPS = permute(allMAPS,[2 1 3 4]);
clims = [0 3;0 1;0 2;0 1;0 3;0 1;0 2;0 1];
nametags = {'MD minimal RICE','FA minimal RICE','MK minimal RICE','$\mu$FA minimal RICE','MD RICE','FA RICE','MK RICE','$\mu$FA RICE'};
figure('Position',[236 281 2096 893]), colormap gray
RICEtools.WrapperPlotManySlices(allMAPS, slice,clims,nametags,2,[],1),

















