%% Example code for RICE (Rotational Invariants of the Cumulant Expansion)
% parameter estimation with the RICE toolbox
%
% EXAMPLE 1 - Fitting full RICE on LTE+PTE data
%
% EXAMPLE 2 - 
%
% EXAMPLE 3 - 
%
%
%% EXAMPLE 1
clc,clear,close all

% Path were the test dataset is located
pathFiles='/Users/coelhs01/Documents/SantiagoCoelho/Git/RICE_example_datasets/dataset_1';

% Load example dataset 1, mask, and protocol
nii_dwi=load_untouch_nii(fullfile(pathFiles,'dwi.nii'));
nii_mask=load_untouch_nii(fullfile(pathFiles,'mask.nii'));
b=load(fullfile(pathFiles,'dwi.bval'));
beta=load(fullfile(pathFiles,'dwi.beta'));
dirs=load(fullfile(pathFiles,'dwi.dirs'));

% Run full RICE fitting (WLLS)
%  - Estimate full D and C tensors from LTE + PTE data
CSphase=1; % Use Condon-Shortley phase in spherical harmonics definition
nsl_flag=1; % Use local nonlinear smoothing for fitting
tic
[b0, tensor_elems, RICE_maps, DIFF_maps] = RICE.fit(nii_dwi.img, b, dirs, beta, logical(nii_mask.img) , CSphase, 'fullRICE', nsl_flag);
% Compute fiber basis projections (axial and radial diffusivities and kurtosis)
DKI_maps = RICE.get_DKI_fiberBasis_maps_from_4D_DW_tensors(tensor_elems, logical(nii_mask.img) , CSphase);
toc

% Plot color FA map
slice=40;
unitBrightness=repmat(DKI_maps.fa(:,:,slice),1,1,3);
MainEigvec_scaled=uint8(abs(squeeze(DKI_maps.fe(:,:,slice,:))).*unitBrightness*255);
figure('Position',[527 467 1033 871]), image(flip(permute(MainEigvec_scaled,[2 1 3]),1)), axis off equal

RICE_maps.W0=3*RICE_maps.S0./(RICE_maps.D0).^2;
RICE_maps.W2=3*RICE_maps.S2./(RICE_maps.D0).^2;
RICE_maps.W4=3*RICE_maps.S4./(RICE_maps.D0).^2;
names={'$D_0$','$D_2$','$W_0$','$W_2$','$W_4$','$A_0$','$A_2$'};
clims=[0 3;0 1.5;0 2;0 1.5;0 0.5;-0.5 0.5;0 1];
allRICE=cat(4,RICE_maps.D0,RICE_maps.D2,RICE_maps.W0,RICE_maps.W2,RICE_maps.W4,RICE_maps.A0,RICE_maps.A2);
figure('Position',[201 232 1971 910]), colormap gray
RICE.WrapperPlotManySlices(permute(allRICE,[2 1 3 4]), slice, clims, names, 2, [], 1, 1),

C2=sqrt(5/(4*pi));
names={'MD','FA','MK','AK','RK','$\mu$FA','SSC'};
clims=[0 3;0 1;0 2;0 1.5;0 3;0 1;0 4];
allDIFF=cat(4,RICE_maps.D0,DKI_maps.fa,DKI_maps.mw,DKI_maps.aw,DKI_maps.rw,DIFF_maps.ufa,C2*DIFF_maps.d0d2m);
figure('Position',[201 232 1971 910]), colormap gray
RICE.WrapperPlotManySlices(permute(allDIFF,[2 1 3 4]), slice, clims, names, 2, [], 1, 1),


