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
%% EXAMPLE 1 - Fitting full RICE on LTE+PTE data. Plotting all RICE maps and
%%%            DKI maps + microscopic FA + size-shape covariance (SSC)
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
nls_flag = 1;       % Use local nonlinear smoothing for fitting to boost SNR
[b0, tensor_elems, RICE_maps, DIFF_maps] = RICE.fit(DWI, b, dirs, beta, mask, CSphase, type, nls_flag);

% Compute fiber basis projections (axial and radial diffusivities and kurtosis)
DKI_maps = RICE.get_DKI_fiberBasis_maps_from_4D_DW_tensors(tensor_elems, mask, CSphase);

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
RICE.WrapperPlotManySlices(permute(allRICE,[2 1 3 4]), slice, clims, names, 2, [], 1, 1),

% Plot DKI maps + microscopic FA + size-shape covariance (SSC)
C2 = sqrt(5/(4*pi));
allDIFF = cat(4,RICE_maps.D0,DKI_maps.fa,DKI_maps.mw,DKI_maps.aw,DKI_maps.rw,DIFF_maps.ufa,C2*DIFF_maps.d0d2m);
clims = [0 3;0 1;0 2;0 1.5;0 3;0 1;0 4];
names = {'MD','FA','MK','AK','RK','$\mu$FA','SSC'};
figure('Position',[201 232 1971 910]), colormap gray
RICE.WrapperPlotManySlices(permute(allDIFF,[2 1 3 4]), slice, clims, names, 2, [], 1, 1),


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

CSphase = 1;  % Use Condon-Shortley phase in spherical harmonics definition
nls_flag = 1; % Use local nonlinear smoothing for fitting to boost SNR

type = 'minimalDKI';  % Estimate minimal DKI, Eq. (30) in RICE paper (WLLS)
[b0_a, tensor_elems_a, RICE_maps_a, DIFF_maps_a] = RICE.fit(DWI_minimalDKI, b_minimalDKI, dirs_minimalDKI, beta_minimalDKI, mask, CSphase, type, nls_flag);

type = 'fullDKI';  % Estimate DKI (WLLS)
[b0_b, tensor_elems_b, RICE_maps_b, DIFF_maps_b] = RICE.fit(DWI_fullDKI, b_fullDKI, dirs_fullDKI, beta_fullDKI, mask, CSphase, type, nls_flag);

% Plot comparison for MD, FA, MK maps
slice = 40;
allMAPS = cat(4,DIFF_maps_a.md,DIFF_maps_a.fa,DIFF_maps_a.mw,DIFF_maps_b.md,DIFF_maps_b.fa,DIFF_maps_b.mw);
allMAPS = permute(allMAPS,[2 1 3 4]);
clims = [0 3;0 1;0 2;0 3;0 1;0 2];
nametags = {'MD minimal DKI','FA minimal DKI','MK minimal DKI','MD DKI','FA DKI','MK DKI'};
figure('Position', [293 304 1683 908]), colormap gray
RICE.WrapperPlotManySlices(allMAPS, slice,clims,nametags,2,[],1),

%% EXAMPLE 3 - Fitting minimal RICE vs full RICE on LTE data.
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

CSphase = 1;  % Use Condon-Shortley phase in spherical harmonics definition
nls_flag = 1; % Use local nonlinear smoothing for fitting to boost SNR

type = 'minimalRICE'; % Estimate minimal RICE, Eq. (31) in RICE paper (WLLS)
[b0_a, tensor_elems_a, RICE_maps_a, DIFF_maps_a] = RICE.fit(DWI_minimalRICE, b_minimalRICE, dirs_minimalRICE, beta_minimalRICE, mask, CSphase, type, nls_flag);

type = 'fullRICE';    % Estimate RICE (WLLS)
[b0_b, tensor_elems_b, RICE_maps_b, DIFF_maps_b] = RICE.fit(DWI_fullRICE, b_fullRICE, dirs_fullRICE, beta_fullRICE, mask, CSphase, type, nls_flag);

% Plot comparison for MD, FA, MK maps
slice = 40;
allMAPS = cat(4,DIFF_maps_a.md,DIFF_maps_a.fa,DIFF_maps_a.mw,DIFF_maps_a.ufa,DIFF_maps_b.md,DIFF_maps_b.fa,DIFF_maps_b.mw,DIFF_maps_b.ufa);
allMAPS = permute(allMAPS,[2 1 3 4]);
clims = [0 3;0 1;0 2;0 1;0 3;0 1;0 2;0 1];
nametags = {'MD minimal RICE','FA minimal RICE','MK minimal RICE','$\mu$FA minimal RICE','MD RICE','FA RICE','MK RICE','$\mu$FA RICE'};
figure('Position',[236 281 2096 893]), colormap gray
RICE.WrapperPlotManySlices(allMAPS, slice,clims,nametags,2,[],1),

















