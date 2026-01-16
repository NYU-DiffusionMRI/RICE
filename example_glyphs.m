%% Example code for RICE (Rotational Invariants of the Cumulant Expansion)
%
% EXAMPLE 4 - Visualizing tensor glyphs - 
%
%% D-C TQ IRREDUCIBLE DECOMPOSITION GLYPHS
clc,clear,close all
load('/Users/coelhs01/Documents/SantiagoCoelho/Git/RICE/tensorfits_example.mat')

RES=100;
[Xs,Ys,Zs]=sphere(RES-1);

% Corpus Callosum voxel
vox_x = 62; vox_y = 68; vox_z = 41; tissue_tag = 'CC';
campos_view = [1.7172   15.0604   16.7235];
cax_lim = [-0.1 0.1]; scaling = [1 1 4 8 15 15 3];

Alm_vox = squeeze(tensor_elems(vox_x,vox_y,vox_z,22:27));
Slm_vox = squeeze(tensor_elems(vox_x,vox_y,vox_z,7:21));
Dlm_vox = squeeze(tensor_elems(vox_x,vox_y,vox_z,1:6));

[Tlm_vox,Qlm_vox] = RICEtools.SA2TQ(Slm_vox, Alm_vox);

Slm_vox = Tlm_vox;
Alm_vox = Qlm_vox;

D00_vox = Dlm_vox; D00_vox(2:6) = 0;
D2m_vox = Dlm_vox; D2m_vox(1) = 0;
A00_vox = Alm_vox; A00_vox(2:6) = 0;
A2m_vox = Alm_vox; A2m_vox(1) = 0;
S2m_vox = Slm_vox(1:6); S2m_vox(1) = 0;
S4m_vox = Slm_vox; S4m_vox(1:6) = 0;
S00_vox = Slm_vox; S00_vox(2:15) = 0;


figure('Position', [2585 572 1812 592]), 

colorMap = [0 0 1;1 0 0];
colormap(colorMap);

[Xr_E1, Yr_E1, Zr_E1, phi1, theta1, Evaluated_E1] = RICEtools.Evaluate_Slm_in_realSHbasis(D00_vox, RES, CSphase);
[Xr_E1sq, Yr_E1sq, Zr_E1sq] = sph2cart(phi1, theta1, abs(Evaluated_E1));
subplot(171), surf(Xr_E1sq, Yr_E1sq, Zr_E1sq,Evaluated_E1);
axis equal off; xlabel 'X', ylabel 'Y', zlabel 'Z'
light; lighting phong, set(gca,'FontSize',20), hold on, shading interp, title('$D^{(0)}$','Interpreter','latex')
campos(campos_view), caxis([-0.7 0.7])

[Xr_E1, Yr_E1, Zr_E1, phi1, theta1, Evaluated_E1] = RICEtools.Evaluate_Slm_in_realSHbasis(D2m_vox, RES, CSphase);
[Xr_E1sq, Yr_E1sq, Zr_E1sq] = sph2cart(phi1, theta1, abs(Evaluated_E1));
subplot(172), surf(Xr_E1sq, Yr_E1sq, Zr_E1sq,Evaluated_E1);
axis equal off; xlabel 'X', ylabel 'Y', zlabel 'Z'
light; lighting phong, set(gca,'FontSize',20), hold on, shading interp, title('$D^{(2)}_{ij}\,n_i n_j $','Interpreter','latex')
campos(campos_view), caxis([-0.7 0.7])

[Xr_E1, Yr_E1, Zr_E1, phi1, theta1, Evaluated_E1] = RICEtools.Evaluate_Slm_in_realSHbasis(S00_vox, RES, CSphase);
[Xr_E1sq, Yr_E1sq, Zr_E1sq] = sph2cart(phi1, theta1, abs(Evaluated_E1));
subplot(173), surf(Xr_E1sq, Yr_E1sq, Zr_E1sq,Evaluated_E1);
axis equal off; xlabel 'X', ylabel 'Y', zlabel 'Z'
light; lighting phong, set(gca,'FontSize',20), hold on, shading interp, title('$T^{(0)}$','Interpreter','latex')
campos(campos_view), caxis([-0.7 0.7])

[Xr_E1, Yr_E1, Zr_E1, phi1, theta1, Evaluated_E1] = RICEtools.Evaluate_Slm_in_realSHbasis(S2m_vox, RES, CSphase);
[Xr_E1sq, Yr_E1sq, Zr_E1sq] = sph2cart(phi1, theta1, abs(Evaluated_E1));
subplot(174), surf(Xr_E1sq, Yr_E1sq, Zr_E1sq,Evaluated_E1);
axis equal off; xlabel 'X', ylabel 'Y', zlabel 'Z'
light; lighting phong, set(gca,'FontSize',20), hold on, shading interp, title('$T^{(2)}_{ij}\,n_i n_j$','Interpreter','latex')
campos(campos_view), caxis([-0.7 0.7])

[Xr_E1, Yr_E1, Zr_E1, phi1, theta1, Evaluated_E1] = RICEtools.Evaluate_Slm_in_realSHbasis(S4m_vox, RES, CSphase);
[Xr_E1sq, Yr_E1sq, Zr_E1sq] = sph2cart(phi1, theta1, abs(Evaluated_E1));
subplot(175), surf(Xr_E1sq, Yr_E1sq, Zr_E1sq,Evaluated_E1);
axis equal off; xlabel 'X', ylabel 'Y', zlabel 'Z'
light; lighting phong, set(gca,'FontSize',20), hold on, shading interp, title('$T^{(4)}_{ijkl}\,n_i n_j n_k n_l$','Interpreter','latex')
campos(campos_view), caxis([-0.7 0.7])

[Xr_E1, Yr_E1, Zr_E1, phi1, theta1, Evaluated_E1] = RICEtools.Evaluate_Slm_in_realSHbasis(A00_vox, RES, CSphase);
[Xr_E1sq, Yr_E1sq, Zr_E1sq] = sph2cart(phi1, theta1, abs(Evaluated_E1));
subplot(176), surf(Xr_E1sq, Yr_E1sq, Zr_E1sq,Evaluated_E1);
axis equal off; xlabel 'X', ylabel 'Y', zlabel 'Z'
light; lighting phong, set(gca,'FontSize',20), hold on, shading interp, title('$Q^{(0)}$','Interpreter','latex')
campos(campos_view), caxis([-0.7 0.7])

[Xr_E1, Yr_E1, Zr_E1, phi1, theta1, Evaluated_E1] = RICEtools.Evaluate_Slm_in_realSHbasis(A2m_vox, RES, CSphase);
[Xr_E1sq, Yr_E1sq, Zr_E1sq] = sph2cart(phi1, theta1, abs(Evaluated_E1));
subplot(177), surf(Xr_E1sq, Yr_E1sq, Zr_E1sq,Evaluated_E1);
axis equal off; xlabel 'X', ylabel 'Y', zlabel 'Z'
light; lighting phong, set(gca,'FontSize',20), hold on, shading interp, title('$Q^{(2)}_{ij}\,n_i n_j $','Interpreter','latex')
campos(campos_view), caxis([-0.7 0.7])

