classdef RICEtools
    % =====================================================================
    % RICE: Rotational Invariants of the Cumulant Expansion (RICE) class
    % This version uses Racah's normalization of spherical harmonics.
    %
    % RICE is an implementation of the generalized cumulant expansion 
    % (any B) framework for representing diffusion MRI signals up to O(b^2)
    %
    % =====================================================================
    % At the moment this class contains some useful tools for
    % estimating tensors, reparametrizing tensors, changing their basis,
    % and also computing their rotational invariants.
    % All tools assume tensors are represented by 2D matrices of size
    % [Nelems x Ntensors]. If inputs are 4D then use 'vectorize' and a
    % mask.
    % 
    % ========================== AVAILABLE TOOLS ==========================
    % - [A_rank2] = Compute_rank2_A_from_rank4_A(A_rank4)
    % - [A_rank4] = Compute_rank4_A_from_rank2_A(A_rank2)
    % - [S_cart15,Csym_cart21] = Symmetrize_C_tensor(C_cart)
    % - [S_cart_21] = Reorganize_S_tensor_15to21_elements(S_cart_15)
    % - [Y_ell] = get_STF_basis(Lmax,CSphase)
    % - [Scart] = STF2cart(Slm,CSphase)
    % - [Slm] = cart2STF(Scart,CSphase)
    % - [Bset,B3x3xN] = ConstructAxiallySymmetricB(b,bshape,dirs)
    % - [Bset_2D] = Generate_BijBkl_2Dset(B_tensors)
    % - [dwi] = nlmsmooth(dwi,mask, akc, smoothlevel)
    % - 
    % - 
    % - 
    % =====================================================================
    %
    %  Authors: Santiago Coelho (santiago.coelho@nyulangone.org), Els Fieremans, Dmitry Novikov
    %  Copyright (c) 2024 New York University
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
    %  REFERENCES:
    %  - Coelho, S., Szczepankiewicz, F., Fieremans, E., Novikov, D.S., 2024 (ArXiv)
    %

    methods     ( Static = true )
        % =================================================================
        function [b0, tensor_elems, RICE_maps, DIFF_maps] = fit(DWI, b, dirs, bshape, mask, CSphase, ComplexSTF, type, nls_flag, parallel_flag)
            % [b0, tensor_elems, RICE_maps, DIFF_maps] = fit(DWI, b, dirs, bshape, mask, CSphase, ComplexSTF, type, nls_flag, parallel_flag)
            %
            % Unconstrained WLLS fit (initialized with LLS) for multiple versions
            % of the cumulant expansion (considering multiple combinations of 
            % isotropic and anisotropic tensors)
            %
            % Compulsory inputs:
            %     - DWI: 4D array of diffusion weighted images [X x Y x Z x Ndwi]
            %     - b: 1D array of b-values
            %     - dirs: [3 x Ndwi] array with normalized diffusion directions
            %             (b-vectors)
            %     - type: string specifying which fitting will be performed:
            %
            %       type can be 'minimalDTI' and then tensor_elem has:   [D00] (1 elem)
            %                                      RICE has: D0
            %                                      maps has: md 
            %       type can be 'fullDTI' and then tensor_elem has:   [D00 D2m] (6 elem)
            %                                      RICE has: D0 D2
            %                                      maps has: md ad rd fa (ad,rd are axial symmetry approximations) 
            %       type can be 'minimalDKI' and then tensor_elem has:   [D00 D2m S00] (7 elem)
            %                                      RICE has: D0 D2 S0
            %                                      maps has: md fa mw
            %       type can be 'minimalDKI_iso' and then tensor_elem has:   [D00 S00] (2 elem)
            %                                      RICE has: D0 S0
            %                                      maps has: md mw 
            %       type can be 'DKI_no_ell4' and then tensor_elem has:   [D00 D2m S00 S2m] (12 elem)
            %                                      RICE has: D0 D2 S0 S2
            %                                      maps has: md ad rd fa mw rw aw (ad,rd,aw,rw are axial symmetry approximations assuming W4=0) 
            %       type can be 'fullDKI' and then tensor_elem has:   [D00 D2m S00 S2m S4m] (21 elem)
            %                                      RICE has: D0 D2 S0 S2 S4
            %                                      maps has: md ad rd fa mw rw aw kfa (ad,rd,aw,rw are axial symmetry approximations) 
            %       type can be 'minimalRICE' and then tensor_elem has:  [D00 D2m S00 S2m A00(rank2)] (14 elem)
            %                                      RICE has: D0 D2 S0 S2 A0
            %                                      maps has: md ad rd fa mw ufa  ](ad,rd are axial symmetry approximations)
            %       type can be 'fullRICE' and then tensor_elem has:  [D00 D2m S00 S2m S4m A00(rank2) A2m(rank2)] (27 elem)
            %                                      RICE has: D0 D2 S0 S2 S4 A0 A2
            %                                      maps has: md ad rd fa mw aw rw kfa ufa d0d2m SSC (ad,rd,aw,rw are axial symmetry approximations) 
            %
            %       maps.ad, maps.rd, maps.aw, and maps.rw are approximations assuming axially symmetric tensors (no need to go to fiber basis for this)
            %
            % Optional inputs:
            %     - bshape: 1D array of b-tensor shapes (assumes axial symmetry).
            %       If absent then it assumes all data is LTE.
            %     - CSphase is a flag to use the Condon-Shortley phase in the
            %       definition of pherical harmonics, default is on.
            %     - mask, binary 3D array. If empty fit is done for all voxels.
            %     - nls_flag, flag specifying if nonlinear smoothing is applied
            %       before the fitting. Default is off.
            %
            % Outputs:
            %     - b0: 3D array with the fit value for b=0
            %     - tensor_elems: explained above under 'type' compulsory input
            %     - RICE: explained above under 'type' compulsory input
            %     - maps: explained above under 'type' compulsory input
            %
            %
            % By: Santiago Coelho (19/10/2022) Santiago.Coelho@nyulangone.org
            % =========================================================================
            if ~exist('parallel_flag','var') || isempty(parallel_flag)
                parallel_flag = true(1);
            end

            [x, y, z, ndwis] = size(DWI);
            if ~exist('mask','var') || isempty(mask)
                mask = true(x, y, z);
            end
            
            if ~exist('bshape','var') || isempty(bshape)
                bshape = ones(size(b));
            end
            
            if ~exist('nls_flag','var') || isempty(nls_flag)
                nls_flag=0;
            end
            
            if nls_flag
                % Default nonlinear smoothing parameters
                smoothlevel=10;
                DWI = RICEtools.nlmsmooth(DWI,mask, 0*mask, smoothlevel);
            end
            
            DWI = RICEtools.vectorize(DWI, mask);
            if size(dirs,2)~=3
                dirs=dirs';
            end
            Nvoxels=size(DWI,2);
            b=b(:);
            Nmeas=length(b);
            if size(dirs,1)~=Nmeas
                error('Number of directions must much number of b-values')
            end
            
            % Evaluate SH on input directions
            if ~exist('CSphase','var') || isempty(CSphase) || CSphase
                CSphase=1; % 1 means we use it
                CS_tag = 'Using';
            else
                CSphase=0; % 0 means we DO NOT use it
                CS_tag = 'Not using';
            end
            if ~exist('ComplexSTF','var') || isempty(ComplexSTF) || ~ComplexSTF
                ComplexSTF=0; % 0 means we use real STF basis (default)
                STF_tag = 'real';
            else
                ComplexSTF=1; % 1 means we use complex STF basis
                STF_tag = 'complex';
                error('fit does not support this option. If you want complex spherical tensor coefficients of cumulant tensors compute them using real basis and then go to complex using eq S25 in RICE paper (Coelho et al. 2024)')
            end
            % Display message to let user know and avoid issues
            fprintf([CS_tag,' the Condon-Shortley phase on the ',STF_tag,' basis spherical harmonics! (Racah normalization)\n'])


            % Defininig STF basis
            Y2 = RICEtools.get_STF_basis(2,CSphase,ComplexSTF,1);
            Y4 = RICEtools.get_STF_basis(4,CSphase,ComplexSTF,1);
            
            % Computing B-tensors
            [Bset,B3x3xN] = RICEtools.ConstructAxiallySymmetricB(b,bshape,dirs);
            [Bset_2D] = RICEtools.Generate_BijBkl_2Dset(Bset);
            Brank=rank(Bset_2D);
            
            % Computing other BijBkl terms
            Bij=reshape(B3x3xN,9,[])';
            BmkBnk=zeros(ndwis,9);
            BijBkl=zeros(ndwis,81);
            for ii=1:ndwis
                BmkBnk(ii,:)=reshape(B3x3xN(:,:,ii)*B3x3xN(:,:,ii),1,[]);
            end
            for id=1:ndwis
                BijBkl_current=zeros(3,3,3,3);
                for ii=1:3
                    for jj=1:3
                        for kk=1:3
                            for ll=1:3
                                BijBkl_current(ii,jj,kk,ll)=B3x3xN(ii,jj,id)*B3x3xN(kk,ll,id);
                            end
                        end
                    end
                end
                BijBkl(id,:)=BijBkl_current(:)';
            end
            BijBij=diag(Bij*Bij');
            delta_mn=reshape(eye(3),1,9);
            BijBkl_epsilon_term=2*(b.^2-BijBij).*delta_mn + 4*BmkBnk - 4*b.*Bij;
            
            % Generating the design matrices
            if strcmp(type,'minimalDTI')
                r=1; % 1 element is fitted: D00
                X=[ones(size(b)), -Bij*Y2(:,1)];
            elseif strcmp(type,'fullDTI')
                r=6; % 2 elements are fitted: D00 D2m
                X=[ones(size(b)), -Bij*Y2];
            elseif strcmp(type,'minimalDKI_iso')
                r=2; % 2 elements are fitted: D00 S00
                X=[ones(size(b)), -Bij*Y2(:,1), 1/2*BijBkl*Y4(:,1)];
            elseif strcmp(type,'minimalDKI')
                r=12; % 7 elements are fitted: D00 D2m S00 (S2m are fitted and discarded)
                X=[ones(size(b)), -Bij*Y2, 1/2*BijBkl*Y4(:,1:6)];
            elseif strcmp(type,'DKI_no_ell4')
                r=12; % 12 elements are fitted: D00 D2m S00 S2m
                X=[ones(size(b)), -Bij*Y2, 1/2*BijBkl*Y4(:,1:6)];
            elseif strcmp(type,'fullDKI')
                r=21; % 21 elements are fitted: D00 D2m S00 S2m S4m
                X=[ones(size(b)), -Bij*Y2, 1/2*BijBkl*Y4];
            elseif strcmp(type,'minimalRICE')
                r=13; % 8 elements are fitted: D00 D2m S00 A00(rank2) (S2m are fitted and discarded)
                X=[ones(size(b)), -Bij*Y2, 1/2*BijBkl*Y4(:,1:6), 1/12*BijBkl_epsilon_term*Y2(:,1)];
            elseif strcmp(type,'fullRICE')
                r=27; % 27 elements are fitted: D00 D2m S00 S2m S4m A00(rank2) A2m(rank2)
                X=[ones(size(b)), -Bij*Y2, 1/2*BijBkl*Y4, 1/12*BijBkl_epsilon_term*Y2];
            else
                error('Choose an appropriate signal representation')
            end    
            if Brank<r, error('rank of protocol B-tensor set is inconsistent with the desired fit'), end
            
            % unconstrained LLS fit
            dv = X\log(DWI);
            w = exp(X*dv);
            % WLLS fit initialized with LLS
            if parallel_flag
                parfor ii = 1:Nvoxels
                    wi = diag(w(:,ii));
                    logdwii = log(DWI(:,ii));
                    dv(:,ii) = (wi*X)\(wi*logdwii);
                end
            else
                for ii = 1:Nvoxels
                    wi = diag(w(:,ii));
                    logdwii = log(DWI(:,ii));
                    dv(:,ii) = (wi*X)\(wi*logdwii);
                end
            end
            
            % Recover b0
            b0 = exp(dv(1,:));
            b0 = RICEtools.vectorize(b0, mask);
            
            % Recover all tensor elements and rotational invariants
            % Racah-Normalized SH
            C0=1;%sqrt(1/(4*pi));
            C2=1;%sqrt(5/(4*pi));
            C4=1;%sqrt(9/(4*pi));       
            tensor_elems=RICEtools.vectorize(dv(2:end,:), mask);
        
            % All approaches extract MD
            D00=tensor_elems(:,:,:,1);
            D0=C0*(D00);
            % Output RICE maps
            RICE_maps.D0=D0;
            % Output Diffusion maps
            DIFF_maps.md=D0;
        
            % If other approaches, then extract other parameters
            if strcmp(type,'fullDTI')
                D2m=tensor_elems(:,:,:,2:6);
                D2=C2*sqrt(sum(D2m.^2,4));
                RICE_maps.D2=D2;
                DIFF_maps.md=D0;
                DIFF_maps.ad=D0+D2;
                DIFF_maps.rd=D0-1/2*D2;
                fa_sq=3*D2.^2./(4*D0.^2+2*D2.^2);
                fa_sq(fa_sq(:)<0)=0;
                DIFF_maps.fa=sqrt(fa_sq);
            elseif strcmp(type,'minimalDKI_iso')
                S00=tensor_elems(:,:,:,2);
                S0=C0*(S00);
                RICE_maps.S0=S0;
                DIFF_maps.md=D0;
                W0=3*S0./D0.^2;
                DIFF_maps.mw=W0;
            elseif strcmp(type,'minimalDKI')
                D2m=tensor_elems(:,:,:,2:6);
                D2=C2*sqrt(sum(D2m.^2,4));
                S00=tensor_elems(:,:,:,7);
                S0=C0*(S00);
                RICE_maps.D2=D2;
                RICE_maps.S0=S0;
                DIFF_maps.md=D0;
                DIFF_maps.ad=D0+D2;
                DIFF_maps.rd=D0-1/2*D2;
                fa_sq=3*D2.^2./(4*D0.^2+2*D2.^2);
                fa_sq(fa_sq(:)<0)=0;
                DIFF_maps.fa=sqrt(fa_sq);
                W0=3*S0./D0.^2;
                DIFF_maps.mw=W0;
            elseif strcmp(type,'DKI_no_ell4')
                D2m=tensor_elems(:,:,:,2:6);
                D2=C2*sqrt(sum(D2m.^2,4));
                S00=tensor_elems(:,:,:,7);
                S0=C0*(S00);
                S2m=tensor_elems(:,:,:,8:12);
                S2=C2*sqrt(sum(S2m.^2,4));
                RICE_maps.D2=D2;
                RICE_maps.S0=S0;
                RICE_maps.S2=S2;
                DIFF_maps.md=D0;
                DIFF_maps.ad=D0+D2;
                DIFF_maps.rd=D0-1/2*D2;
                fa_sq=3*D2.^2./(4*D0.^2+2*D2.^2);
                fa_sq(fa_sq(:)<0)=0;
                DIFF_maps.fa=sqrt(fa_sq);
                W0=3*S0./D0.^2;
                W2=3*S2./D0.^2;
                DIFF_maps.mw=W0;
                DIFF_maps.aw=(W0+W2).*D0.^2./(D0+D2).^2;
                DIFF_maps.rw=(W0-1/2*W2).*D0.^2./(D0-1/2*D2).^2;
            elseif strcmp(type,'fullDKI')
                D2m=tensor_elems(:,:,:,2:6);
                D2=C2*sqrt(sum(D2m.^2,4));
                S00=tensor_elems(:,:,:,7);
                S0=C0*(S00);
                S2m=tensor_elems(:,:,:,8:12);
                S2=C2*sqrt(sum(S2m.^2,4));
                S4m=tensor_elems(:,:,:,13:21);
                S4=C4*sqrt(sum(S4m.^2,4));
                RICE_maps.D2=D2;
                RICE_maps.S0=S0;
                RICE_maps.S2=S2;
                RICE_maps.S4=S4;
                DIFF_maps.md=D0;
                DIFF_maps.ad=D0+D2;
                DIFF_maps.rd=D0-1/2*D2;
                fa_sq=3*D2.^2./(4*D0.^2+2*D2.^2);
                fa_sq(fa_sq(:)<0)=0;
                DIFF_maps.fa=sqrt(fa_sq);
                W0=3*S0./D0.^2;
                W2=3*S2./D0.^2;
                W4=3*S4./D0.^2;
                DIFF_maps.mw=W0;
                DIFF_maps.aw=(W0+W2+W4).*D0.^2./(D0+D2).^2;
                DIFF_maps.rw=(W0-1/2*W2+3/8*W4).*D0.^2./(D0-1/2*D2).^2;
                kfa_sq = (14*W2.^2+35*W4.^2)./(40*W0.^2+14*W2.^2+35*W4.^2);
                kfa_sq(kfa_sq(:)<0)=0;
                DIFF_maps.kfa = sqrt(kfa_sq);
            elseif strcmp(type,'minimalRICE')
                D2m=tensor_elems(:,:,:,2:6);
                D2=C2*sqrt(sum(D2m.^2,4));
                S00=tensor_elems(:,:,:,7);
                S0=C0*(S00);
                S2m=tensor_elems(:,:,:,8:12);
                S2=C2*sqrt(sum(S2m.^2,4));
                A00=tensor_elems(:,:,:,13);
                A0=C0*A00;
                % Isotropic part of A_{ijkl}=A0*2/5;
                RICE_maps.D2=D2;
                RICE_maps.S0=S0;
                RICE_maps.S2=S2;
                RICE_maps.A0=A0;
                DIFF_maps.md=D0;
                DIFF_maps.ad=D0+D2;
                DIFF_maps.rd=D0-1/2*D2;
                fa_sq=3*D2.^2./(4*D0.^2+2*D2.^2);
                fa_sq(fa_sq(:)<0)=0;
                DIFF_maps.fa=sqrt(fa_sq);
                W0=3*S0./D0.^2;
                DIFF_maps.mw=W0;
                % Other maps
                T0 = 4/9 * S0 - 2/9 * A0;
                Q0 = 5/9 * S0 + 2/9 * A0;
%                 ufa_sq=(3*T0 + 3*D2.^2)./(2*T0 + 2*D2.^2 + 4*D0.^2 );
                ufa_sq=(15*T0 + 3*D2.^2)./(10*T0 + 2*D2.^2 + 4*D0.^2 );
                ufa_sq(ufa_sq(:)<0)=0;
                DIFF_maps.ufa=sqrt(ufa_sq);
            elseif strcmp(type,'fullRICE')
                D2m=tensor_elems(:,:,:,2:6);
                D2=C2*sqrt(sum(D2m.^2,4));
                S00=tensor_elems(:,:,:,7);
                S0=C0*(S00);
                S2m=tensor_elems(:,:,:,8:12);
                S2=C2*sqrt(sum(S2m.^2,4));
                S4m=tensor_elems(:,:,:,13:21);
                S4=C4*sqrt(sum(S4m.^2,4));
                A00=tensor_elems(:,:,:,22);
                A0=C0*A00;
                A2m=tensor_elems(:,:,:,23:27);
                A2=C2*sqrt(sum(A2m.^2,4));
                % Isotropic part of A_{ijkl}=A0*2/5;
                RICE_maps.D2=D2;
                RICE_maps.S0=S0;
                RICE_maps.S2=S2;
                RICE_maps.S4=S4;
                RICE_maps.A0=A0;
                RICE_maps.A2=A2;
                DIFF_maps.md=D0;
                DIFF_maps.ad=D0+D2;
                DIFF_maps.rd=D0-1/2*D2;
                fa_sq=3*D2.^2./(4*D0.^2+2*D2.^2);
                fa_sq(fa_sq(:)<0)=0;
                DIFF_maps.fa=sqrt(fa_sq);
                W0=3*S0./D0.^2;
                W2=3*S2./D0.^2;
                W4=3*S4./D0.^2;
                DIFF_maps.mw=W0;
                DIFF_maps.aw=(W0+W2+W4).*D0.^2./(D0+D2).^2;
                DIFF_maps.rw=(W0-1/2*W2+3/8*W4).*D0.^2./(D0-1/2*D2).^2;
                kfa_sq = (14*W2.^2+35*W4.^2)./(40*W0.^2+14*W2.^2+35*W4.^2);
                kfa_sq(kfa_sq(:)<0)=0;
                DIFF_maps.kfa = sqrt(kfa_sq);
                % Other maps
                T0 = 4/9 * S0 - 2/9 * A0;
                Q0 = 5/9 * S0 + 2/9 * A0;
                Q2m = 1/9 * (7 * S2m - 2 * A2m);
%                 ufa_sq=(3*T0 + 3*D2.^2)./(2*T0 + 2*D2.^2 + 4*D0.^2 );
                ufa_sq=(15*T0 + 3*D2.^2)./(10*T0 + 2*D2.^2 + 4*D0.^2 );
%                 T0 = 4/7 * S0 - 2/7 * A0;
%                 Q0 = 3/7 * S0 + 2/7 * A0;
%                 Q2m = 1/9 * (7 * S2m - 2 * A2m);
                Q2=C2*sqrt(sum(Q2m.^2,4));
                ufa_sq(ufa_sq(:)<0)=0;
                DIFF_maps.ufa=sqrt(ufa_sq);
                SSC=1/2 * Q2./sqrt(abs(5*Q0.*T0));
                DIFF_maps.SSC=SSC;
                % DIFF_maps.d0d2m_doubleBra = 1/2 * Q2m;
            end    
        end    
        % =================================================================  
        function [S_cart_15,Csym_cart_21] = Symmetrize_C_tensor(C_cart)
            % [S_cart_15,Csym_cart_21] = Symmetrize_C_tensor(C_cart)
            %
            % C: 2D array containing [C_1111;C_2222;C_3333;C_1212;C_1313;C_2323;C_1122;...
            %                         C_1133;C_1112;C_1113;C_1123;C_2233;C_2212;C_2213;...
            %                         C_2223;C_3312;C_3313;C_3323;C_1213;C_1223;C_1323];
            % S: 2D array containing [S_1111;S_2222;S_3333;S_1122;S_1133;...
            %                         S_1112;S_1113;S_1123;S_2233;S_2212;...
            %                         S_2213;S_2223;S_3312;S_3313;S_3323];   
            % Csym:  2D array containing the repeated elements of S in the C's format
            S_cart_15=C_cart(1:15,:)*0;
            S_cart_15(1,:)=C_cart(1,:);
            S_cart_15(2,:)=C_cart(2,:);
            S_cart_15(3,:)=C_cart(3,:);
            S_cart_15(4,:)=1/3*C_cart(7,:)+2/3*C_cart(4,:);
            S_cart_15(5,:)=1/3*C_cart(8,:)+2/3*C_cart(5,:);
            S_cart_15(6,:)=C_cart(9,:);
            S_cart_15(7,:)=C_cart(10,:);
            S_cart_15(8,:)=1/3*C_cart(11,:)+2/3*C_cart(19,:);
            S_cart_15(9,:)=1/3*C_cart(12,:)+2/3*C_cart(6,:);
            S_cart_15(10,:)=C_cart(13,:);
            S_cart_15(11,:)=1/3*C_cart(14,:)+2/3*C_cart(20,:);
            S_cart_15(12,:)=C_cart(15,:);
            S_cart_15(13,:)=1/3*C_cart(16,:)+2/3*C_cart(21,:);
            S_cart_15(14,:)=C_cart(17,:);
            S_cart_15(15,:)=C_cart(18,:);
            Csym_cart_21=C_cart*0;
            Csym_cart_21(1,:)=S_cart_15(1,:);
            Csym_cart_21(2,:)=S_cart_15(2,:);
            Csym_cart_21(3,:)=S_cart_15(3,:);
            Csym_cart_21(4,:)=S_cart_15(4,:);%4-
            Csym_cart_21(5,:)=S_cart_15(5,:);%5-
            Csym_cart_21(6,:)=S_cart_15(9,:);%9-
            Csym_cart_21(7,:)=S_cart_15(4,:);%4-
            Csym_cart_21(8,:)=S_cart_15(5,:);%5-
            Csym_cart_21(9,:)=S_cart_15(6,:);
            Csym_cart_21(10,:)=S_cart_15(7,:);
            Csym_cart_21(11,:)=S_cart_15(8,:);%8-
            Csym_cart_21(12,:)=S_cart_15(9,:);%9-
            Csym_cart_21(13,:)=S_cart_15(10,:);
            Csym_cart_21(14,:)=S_cart_15(11,:);%11-
            Csym_cart_21(15,:)=S_cart_15(12,:);
            Csym_cart_21(16,:)=S_cart_15(13,:);%13-
            Csym_cart_21(17,:)=S_cart_15(14,:);
            Csym_cart_21(18,:)=S_cart_15(15,:);
            Csym_cart_21(19,:)=S_cart_15(8,:);%8-
            Csym_cart_21(20,:)=S_cart_15(11,:);%11-
            Csym_cart_21(21,:)=S_cart_15(13,:);%13-
        end
        % =================================================================
        function [S_cart_21] = Reorganize_S_tensor_15to21_elements(S_cart_15)
            % [S_cart_21] = Reorganize_S_tensor_15to21_elements(S_cart_15)
            %
            % S_cart_15: 2D array containing [S_1111;S_2222;S_3333;S_1122;S_1133;...
            %                                 S_1112;S_1113;S_1123;S_2233;S_2212;...
            %                                 S_2213;S_2223;S_3312;S_3313;S_3323];   
            %
            % S_cart_21: 2D array containing [S_1111;S_2222;S_3333;S_1212;S_1313;S_2323;S_1122;...
            %                                 S_1133;S_1112;S_1113;S_1123;S_2233;S_2212;S_2213;...
            %                                 S_2223;S_3312;S_3313;S_3323;S_1213;S_1223;S_1323];
            %
            % S_cart_21:  2D array containing the repeated elements of S_cart_15 in the C's format
            Nvox=size(S_cart_15,2);
            S_cart_21=zeros(21,Nvox);
            S_cart_21(1,:)=S_cart_15(1,:);
            S_cart_21(2,:)=S_cart_15(2,:);
            S_cart_21(3,:)=S_cart_15(3,:);
            S_cart_21(4,:)=S_cart_15(4,:);%4-
            S_cart_21(5,:)=S_cart_15(5,:);%5-
            S_cart_21(6,:)=S_cart_15(9,:);%9-
            S_cart_21(7,:)=S_cart_15(4,:);%4-
            S_cart_21(8,:)=S_cart_15(5,:);%5-
            S_cart_21(9,:)=S_cart_15(6,:);
            S_cart_21(10,:)=S_cart_15(7,:);
            S_cart_21(11,:)=S_cart_15(8,:);%8-
            S_cart_21(12,:)=S_cart_15(9,:);%9-
            S_cart_21(13,:)=S_cart_15(10,:);
            S_cart_21(14,:)=S_cart_15(11,:);%11-
            S_cart_21(15,:)=S_cart_15(12,:);
            S_cart_21(16,:)=S_cart_15(13,:);%13-
            S_cart_21(17,:)=S_cart_15(14,:);
            S_cart_21(18,:)=S_cart_15(15,:);
            S_cart_21(19,:)=S_cart_15(8,:);%8-
            S_cart_21(20,:)=S_cart_15(11,:);%11-
            S_cart_21(21,:)=S_cart_15(13,:);%13-
        end
        % =================================================================  
        function [A_rank4] = Compute_rank4_A_from_rank2_A(A_rank2)
            % [A_rank4] = Compute_rank4_A_from_rank2_A(A_rank2)
            %
            % A_rank2: 2D array containing [A_11;A_22;A_33;A_12;A_13;A_23]
            %
            % A_rank4: 2D array containing [A_1111;A_2222;A_3333;A_1212;A_1313;A_2323;A_1122;...
            %                               A_1133;A_1112;A_1113;A_1123;A_2233;A_2212;A_2213;...
            %                               A_2223;A_3312;A_3313;A_3323;A_1213;A_1223;A_1323];
            trA = A_rank2(1,:) + A_rank2(2,:) + A_rank2(3,:);
            idx=[ 1,1,1,1; 2,2,2,2; 3,3,3,3; 1,2,1,2; 1,3,1,3; 2,3,2,3; 1,1,2,2;...
                  1,1,3,3; 1,1,1,2; 1,1,1,3; 1,1,2,3; 2,2,3,3; 2,2,1,2; 2,2,1,3;...
                  2,2,2,3; 3,3,1,2; 3,3,1,3; 3,3,2,3; 1,2,1,3; 1,2,2,3; 1,3,2,3];
            delta=eye(3);
            A_rank2_3D(1,1,:)=A_rank2(1,:);
            A_rank2_3D(1,2,:)=A_rank2(4,:);
            A_rank2_3D(1,3,:)=A_rank2(5,:);
            A_rank2_3D(2,1,:)=A_rank2(4,:);
            A_rank2_3D(2,2,:)=A_rank2(2,:);
            A_rank2_3D(2,3,:)=A_rank2(6,:);
            A_rank2_3D(3,1,:)=A_rank2(5,:);
            A_rank2_3D(3,2,:)=A_rank2(6,:);
            A_rank2_3D(3,3,:)=A_rank2(3,:);
            for id=1:size(idx,1)
                ii=idx(id,1); jj=idx(id,2); kk=idx(id,3); ll=idx(id,4);
                term1(id,:) = 2*delta(ii,jj)*delta(kk,ll) - delta(ii,ll)*delta(jj,kk) - delta(ii,kk)*delta(jj,ll);
                term2(id,:) = delta(ii,jj)*A_rank2_3D(kk,ll,:) + delta(kk,ll)*A_rank2_3D(ii,jj,:);
                term3(id,:) = delta(jj,kk)*A_rank2_3D(ii,ll,:) + delta(ii,ll)*A_rank2_3D(jj,kk,:) + delta(ii,kk)*A_rank2_3D(jj,ll,:) + delta(jj,ll)*A_rank2_3D(ii,kk,:);
            end
            A_rank4 = 1/6*( trA.*term1 - 2*term2 + term3 );
        end
        % =================================================================  
        function [A_rank2] = Compute_rank2_A_from_rank4_A(A_rank4)
            % [A_rank2] = Compute_rank2_A_from_rank4_A(A_rank4)
            %
            % A_rank4: 2D array containing [A_1111;A_2222;A_3333;A_1212;A_1313;A_2323;A_1122;...
            %                               A_1133;A_1112;A_1113;A_1123;A_2233;A_2212;A_2213;...
            %                               A_2223;A_3312;A_3313;A_3323;A_1213;A_1223;A_1323];
            %
            % A_rank2: 2D array containing [A_11;A_22;A_33;A_12;A_13;A_23]
            A_rank2=A_rank4(1:6,:)*0;
            A_rank2(1,:) = 2*( A_rank4(12,:) - A_rank4(6,:) );
            A_rank2(2,:) = 2*( A_rank4(8,:) - A_rank4(5,:) );
            A_rank2(3,:) = 2*( A_rank4(7,:) - A_rank4(4,:) );
            A_rank2(4,:) = 2*( A_rank4(21,:) - A_rank4(16,:) );
            A_rank2(5,:) = 2*( A_rank4(20,:) - A_rank4(14,:) );
            A_rank2(6,:) = 2*( A_rank4(19,:) - A_rank4(11,:) );
        end
        % =================================================================
        function [Bset,B3x3xN] = ConstructAxiallySymmetricB(b,bshape,dirs)
            % [Bset,B3x3xN] = ConstructAxiallySymmetricB(b,bshape,dirs)
            Ndwi=length(b);
            Bset=zeros(6,Ndwi);
            if size(dirs,1)~=3
                dirs=dirs';
            end
            if size(dirs,2)~=Ndwi || length(bshape)~=Ndwi
                error('dirs should have the same number of columns as elements in b and bshape')
            end
            for ii=1:Ndwi
                B_aux= bshape(ii)*dirs(:,ii)*dirs(:,ii)' + 1/3*(1-bshape(ii))* eye(3);
                B_aux=B_aux/trace(B_aux);
                if any(isnan(B_aux(:)))
                    B_aux=zeros(3);
                else
                    B_aux=b(ii)*B_aux;
                end
                Bset(:,ii)=[B_aux(1,1),B_aux(2,2),B_aux(3,3),B_aux(1,2),B_aux(1,3),B_aux(2,3)]';
                B3x3xN(:,:,ii)=B_aux;
            end
        end
        % =================================================================
        function [Bset_2D] = Generate_BijBkl_2Dset(B_tensors)
            % [Bset_2D] = Generate_BijBkl_2Dset(B_tensors)
            %
            % Inputs: B_tensors: matrix of b-tensors (diffusion weightings in ms/um^2)
            %         [Nx6] [Bxx,Byy,Bzz,Bxy,Bxz,Byz].
            %         These are NOT normalised, the trace of the b-tensor is the
            %         b-value.
            %
            % Important: Input must be in b-tensor space (not U, its Cholesky decomposition) 
            %
            % A general Btensor has the form:
            % [B(1,1);B(1,2);B(1,3);B(2,2);B(2,3);B(3,3)] (it is a 3x3 matrix)
            %
            % This function computes the corresponding set of b-tensors
            % [3x3xN] to a matrix of dimensions [Nx27] (for D+Z estimation)
            %
            % B_set_2D=[-[B11,B22,B33,2*B12,2*B13,2*B23],...
            %        1/2*[B11.*B11,B22.*B22,B33.*B33,4*B12.*B12,4*B13.*B13,4*B23.*B23,...
            %           2*B11.*B22,2*B11.*B33,4*B11.*B12,4*B11.*B13,4*B11.*B23,...
            %           2*B22.*B33,4*B22.*B12,4*B22.*B13,4*B22.*B23,...
            %           4*B33.*B12,4*B33.*B13,4*B33.*B23,...
            %           8*B12.*B13,8*B12.*B23,...
            %           8*B13.*B23]];
            %
            if size(B_tensors,2)~=6
                B_tensors=B_tensors';
            end
            b=sum(B_tensors(:,1:3),2);
            B11=B_tensors(:,1);
            B22=B_tensors(:,2);
            B33=B_tensors(:,3);
            B12=B_tensors(:,4);
            B13=B_tensors(:,5);
            B23=B_tensors(:,6);
            Bset_2D=[-[B11,B22,B33,2*B12,2*B13,2*B23],...
                  1/2*[B11.*B11,B22.*B22,B33.*B33,4*B12.*B12,4*B13.*B13,4*B23.*B23,...
                     2*B11.*B22,2*B11.*B33,4*B11.*B12,4*B11.*B13,4*B11.*B23,...
                     2*B22.*B33,4*B22.*B12,4*B22.*B13,4*B22.*B23,...
                     4*B33.*B12,4*B33.*B13,4*B33.*B23,...
                     8*B12.*B13,8*B12.*B23,...
                     8*B13.*B23]];
            b0_flag=b<0.01;
            Bset_2D(b0_flag,:)=0;
        end
        % =================================================================
        function [dwi] = nlmsmooth(dwi,mask, akc, smoothlevel)
            % dwi = nlmsmooth(dwi,mask, akc, smoothlevel)
            % define the size of a nonlocal patch
            kernel = 5;
            k = floor(kernel/2);
            k_ = ceil(kernel/2);
            % mask to indicate which voxels should be smoothed
            maskinds = find(mask);
            % normalize contrast for the dwi (for computing similarity only)
            dwi_norm = zeros(size(dwi));
            for i = 1:size(dwi,4)
                dwii = abs(dwi(:,:,:,i));
                dwi_norm(:,:,:,i) = dwii./max(dwii(:));
            end
            % nonlocal smoothing loop
            dwi_ = zeros(length(maskinds),size(dwi,4));
            parfor index = 1 : length(maskinds)
                % grab the index for the voxel we are smoothing
                thisLinearIndex = maskinds(index);
                [x,y,z] = ind2sub(size(mask), thisLinearIndex);
        
                if x-k < 1,           xmin = 1; else; xmin = x-k; end
                if x+k > size(dwi,1), xmax = size(dwi,1); else, xmax = x+k; end
                if y-k < 1,           ymin = 1; else; ymin = y-k; end
                if y+k > size(dwi,2), ymax = size(dwi,2); else, ymax = y+k; end
                if z-k < 1,           zmin = 1; else; zmin = z-k; end
                if z+k > size(dwi,3), zmax = size(dwi,3); else, zmax = z+k; end
                psize = length(xmin:xmax)*length(ymin:ymax)*length(zmin:zmax);
        
                % patch for outlier mask
                akcpatch = reshape(logical(akc(xmin:xmax,ymin:ymax,zmin:zmax)),[psize,1]);
                % patch for center voxel
                ref = repmat(reshape(dwi_norm(x,y,z,:),[1,size(dwi,4)]),[psize,1]);
                % normalized patch
                patch = reshape(dwi_norm(xmin:xmax,ymin:ymax,zmin:zmax,:),[psize,size(dwi,4)]);
                % patch of original dwi data
                patchorig = reshape(dwi(xmin:xmax,ymin:ymax,zmin:zmax,:),[psize,size(dwi,4)]);
                % compute the similary between center voxel adn the rest of the
                % patch summed over all directions
                intensities = sqrt(sum((patch-ref).^2,2))./size(dwi,4);
        
                [min_wgs,min_idx] = sort(intensities, 'ascend');
                wgs_max = min_wgs(end);
                % outliers have low similarity
                min_wgs(akcpatch) = wgs_max;
        
                % threshold similarity map to some percentile of the pach
                goodidx = min_wgs < prctile(min_wgs,smoothlevel);
                min_idx = min_idx(goodidx);
                min_wgs = min_wgs(goodidx);
                wgs_max = max(min_wgs);
        
                % normalize the weights and compute weighted sum
                wgs_inv = wgs_max - min_wgs;
                wgs_nrm = wgs_inv/sum(wgs_inv);
                wval = sum(patchorig(min_idx,:).*(wgs_nrm*ones(1,size(dwi,4))));
                dwi_(index,:)= wval;
            end
             % replace original voxels with smoothed voxels
             for idx = 1:length(maskinds)
                thisLinearIndex = maskinds(idx);
                [x,y,z] = ind2sub(size(mask),thisLinearIndex);
                dwi(x,y,z,:) = dwi_(idx,:);
             end
        end
        % =================================================================
        function Ylm_n = evaluate_even_SH(dirs, Lmax, CS_phase, ComplexSTF)
            % Ylm_n = get_even_SH(dirs,Lmax,CS_phase, ComplexSTF)
            %
            % if CS_phase=1, then the definition uses the Condon-Shortley phase factor
            % of (-1)^m. Default is CS_phase=0 (so this factor is ommited)
            %
            % By: Santiago Coelho (10/06/2021)
            
            
            if size(dirs,2)~=3
                dirs=dirs';
            end

            if ~exist('ComplexSTF','var') || isempty(ComplexSTF) || ~ComplexSTF
                ComplexSTF=0; % 0 means we use real STF basis (default)
            else
                ComplexSTF=1; % 1 means we use complex STF basis
            end
            if ~exist('CS_phase','var') || isempty(CS_phase) || ~CS_phase
                CS_phase=0; % 0 means we do not use Condon-Shortley phase (default)
            else
                CS_phase=1; % 1 means we use Condon-Shortley phase (should be used for complex SH)
            end

            Nmeas=size(dirs,1);
            [PHI,THETA,~]=cart2sph(dirs(:,1),dirs(:,2),dirs(:,3)); THETA=pi/2-THETA;
            l=0:2:Lmax;
            l_all=[];
            m_all=[];
            for ii=1:length(l)
                l_all=[l_all, l(ii)*ones(1,2*l(ii)+1)];
                m_all=[m_all -l(ii):l(ii)];
            end
            % K_lm=sqrt((2*l_all+1)./(4*pi) .* factorial(l_all-abs(m_all))./factorial(l_all+abs(m_all)));
            % Using Racah normalization
            K_lm=sqrt(factorial(l_all-abs(m_all))./factorial(l_all+abs(m_all)));

            P_l_in_cos_theta=zeros(length(l_all),Nmeas);
            if ~ComplexSTF % Use real definition of SH
                if ~CS_phase
                    extra_factor=ones(size(K_lm));
                    extra_factor(m_all~=0)=sqrt(2);
                else
                    extra_factor=ones(size(K_lm));
                    extra_factor(m_all~=0)=sqrt(2);
                    extra_factor=extra_factor.*(-1).^(m_all);
                end
                phi_term=zeros(length(l_all),Nmeas);
                id_which_pl=zeros(1,length(l_all));
                for ii=1:length(l_all)
                    all_Pls=legendre(l_all(ii),cos(THETA));
                    P_l_in_cos_theta(ii,:)=all_Pls(abs(m_all(ii))+1,:);
                    id_which_pl(ii)=abs(m_all(ii))+1;
                    if m_all(ii)>0
                        phi_term(ii,:)=cos(m_all(ii)*PHI);
                    elseif m_all(ii)==0
                        phi_term(ii,:)=1;
                    elseif m_all(ii)<0
                        phi_term(ii,:)=sin(-m_all(ii)*PHI);
                    end
                end
            else % Use complex definition of SH
                if ~CS_phase
                    extra_factor=ones(size(K_lm));
                else
                    extra_factor=ones(size(K_lm));
                    extra_factor=extra_factor.*(-1).^abs(m_all);
                end
                phi_term=zeros(length(l_all),Nmeas);
                id_which_pl=zeros(1,length(l_all));
                for ii=1:length(l_all)
                    all_Pls=legendre(l_all(ii),cos(THETA));
                    if m_all(ii)<0
                        phase_plm = (-1)^abs(m_all(ii)) ;
                    else
                        phase_plm = 1;
                    end
                    P_l_in_cos_theta(ii,:)=phase_plm * all_Pls(abs(m_all(ii))+1,:);
                    id_which_pl(ii)=abs(m_all(ii))+1;
                    phi_term(ii,:)=exp(1i*m_all(ii)*PHI);
                end
            end
            Y_lm=repmat(extra_factor',1,Nmeas).*repmat(K_lm',1,Nmeas).*phi_term.*P_l_in_cos_theta;
            Ylm_n=transpose(Y_lm);
        end
        % =================================================================
        function DKI_maps = get_DKI_fiberBasis_maps_from_4D_DW_tensors(dt, mask, CSphase, ComplexSTF)
            % DKI_maps = get_DKI_fiberBasis_maps_from_4D_DW_tensors(dt, mask, CSphase)
            %
            % dt contains [dlm, slm]
            %
            % DKI_maps contains:
            % - DTI scalar maps: md fa ad_axsym rd_axsym ad rd
            % - DTI first eigenvector: fe
            % - DKI scalar maps: mw aw_axsym rw_axsym rw aw mk rk ak
            %
            %
            % By: Santiago Coelho (07/09/2021) Santiago.Coelho@nyulangone.org
            [ x, y, z, ~] = size(dt);
            if ~exist('mask','var') || isempty(mask)
                mask = true(x, y, z);
            end
            if ~exist('CSphase','var') || isempty(CSphase)
                CSphase=1;
            else
                CSphase=0;
            end
            if ~exist('ComplexSTF','var') || isempty(ComplexSTF) || ~ComplexSTF
                ComplexSTF=0; % 0 means we use real STF basis (default)
            else
                ComplexSTF=1; % 1 means we use complex STF basis
            end
            dt=double(dt);
            dt(~isfinite(dt(:)))=0;
            C0=1;%sqrt(1/(4*pi));
            C2=1;%sqrt(5/(4*pi));
            C4=1;%sqrt(9/(4*pi));   

            % Transform Slm into Wlm (they are proportional)
            md=C0*(dt(:,:,:,1));
            dt_stf=RICEtools.vectorize(cat(4,dt(:,:,:,1:6),3*dt(:,:,:,7:21)./md.^2),mask);
            MDSq=(C0*dt_stf(1,:)).^2;  

            % Compute Diffusion and Kurtosis rotational invariants
            D00=dt(:,:,:,1);
            D0=C0*(D00);
            D2m=dt(:,:,:,2:6);
            D2=C2*sqrt(sum(D2m.^2,4));
            S00=dt(:,:,:,7);
            S0=C0*(S00);
            S2m=dt(:,:,:,8:12);
            S2=C2*sqrt(sum(S2m.^2,4));
            S4m=dt(:,:,:,13:21);
            S4=C4*sqrt(sum(S4m.^2,4));   
            W0=3*S0./D0.^2;
            W2=3*S2./D0.^2;
            W4=3*S4./D0.^2;

            % Get DTI scalars without projecting on fiber basis
            DKI_maps.md=D0;
            DKI_maps.ad_axsym=D0+D2;
            DKI_maps.rd_axsym=D0-1/2*D2;
            fa_sq=3*D2.^2./(4*D0.^2+2*D2.^2);
            fa_sq(fa_sq(:)<0)=0;
            DKI_maps.fa=sqrt(fa_sq);
        
            % Get DKI scalars without projecting on fiber basis
            DKI_maps.mw=W0;
            DKI_maps.aw_axsym=(W0+W2+W4).*D0.^2./(D0+D2).^2;
            DKI_maps.rw_axsym=(W0-1/2*W2+3/8*W4).*D0.^2./(D0-1/2*D2).^2;
        
            % Getting MK
            dirs256 = RICEtools.get256dirs();
            Y_LM_matrix256 = RICEtools.evaluate_even_SH(dirs256,4,CSphase);
            adc = Y_LM_matrix256(:,1:6)*dt_stf(1:6,:);
            akc = (Y_LM_matrix256*dt_stf(7:21,:)).*MDSq./(adc.^2);
            DKI_maps.mk=RICEtools.vectorize(mean(akc,1),mask);

            % Computing AW and RW with their exact definitions an all K maps
            Y2_STF = RICEtools.get_STF_basis(2,CSphase,ComplexSTF,0);
            Y00 = Y2_STF(:,:,1);
            Y2m2 = Y2_STF(:,:,2);
            Y2m1 = Y2_STF(:,:,3);
            Y20 = Y2_STF(:,:,4);
            Y21 = Y2_STF(:,:,5);
            Y22 = Y2_STF(:,:,6);

            wpar=0*dt_stf(1, :);
            wperp=0*dt_stf(1, :);
            rk=0*dt_stf(1, :);
            e1=0*dt_stf(1:3, :);
            lambda=0*dt_stf(1:3, :);
            parfor ii = 1:size(dt_stf,2)
                dt_current=dt_stf(:,ii);
                Dlm=dt_current(1:6);
                Wlm=dt_current(7:21);
                DT = Dlm(1)*Y00 + Dlm(2)*Y2m2 + Dlm(3)*Y2m1 + Dlm(4)*Y20 + Dlm(5)*Y21 + Dlm(6)*Y22;
                [eigvec, eigval] = eigs(real(DT));
                e1(:, ii) = eigvec(:, 1);
                lambda(:,ii) = diag(eigval);
                % Getting AW(AK), RW, and RK
                Ylm_e1 = RICEtools.evaluate_even_SH(eigvec(:, 1),4,CSphase,ComplexSTF);
                wpar(ii)=Ylm_e1*Wlm;
                dirs_radial = RICEtools.radialsampling(eigvec(:, 1), 100);
                Ylm_radial = RICEtools.evaluate_even_SH(dirs_radial,4,CSphase,ComplexSTF);
                D_n_radial = Ylm_radial(:,1:6)*Dlm;
                W_n_radial=Ylm_radial*Wlm;
                wperp(ii)=mean(W_n_radial,1);
        %         rk(ii)=mean(Ylm_radial(:,1:15)*wlm(1:15,ii)*MDSq(ii)/((l2(ii)/2+l3(ii)/2)^2));
                rk(ii)=MDSq(ii)*mean(W_n_radial./D_n_radial.^2,1);
            end
            ad = RICEtools.vectorize(lambda(1,:), mask);
            rd = RICEtools.vectorize(sum(lambda(2:3,:),1)/2, mask);
            wpar = RICEtools.vectorize(wpar, mask);
            wperp = RICEtools.vectorize(wperp, mask);
            DKI_maps.rd=rd;
            DKI_maps.ad=ad;
            DKI_maps.rw=wperp.*D0.^2./rd.^2;
            DKI_maps.aw=wpar.*D0.^2./ad.^2;
            DKI_maps.rk=RICEtools.vectorize(rk,mask);
            DKI_maps.ak=DKI_maps.aw; % They are the same
            DKI_maps.fe=RICEtools.vectorize(e1,mask);
        end
        % =================================================================
        function e1 = get_fiberBasis_from_4D_D_tensor(dt, mask, CSphase, ComplexSTF)
            % e1 = get_fiberBasis_from_4D_D_tensor(dt, mask, CSphase, ComplexSTF)
            %
            % dt contains dlm
            %
            % e1 contains:
            % - first eigenvector: fe
            %
            %
            % By: Santiago Coelho (07/09/2021) Santiago.Coelho@nyulangone.org
            [ x, y, z, ~] = size(dt);
            if ~exist('mask','var') || isempty(mask)
                mask = true(x, y, z);
            end
            if ~exist('CSphase','var') || isempty(CSphase)
                CSphase=1;
            else
                CSphase=0;
            end
            if ~exist('ComplexSTF','var') || isempty(ComplexSTF) || ~ComplexSTF
                ComplexSTF=0; % 0 means we use real STF basis (default)
            else
                ComplexSTF=1; % 1 means we use complex STF basis
            end
            dt=double(dt);
            dt(~isfinite(dt(:)))=0;
            dt_stf=RICEtools.vectorize(dt(:,:,:,1:6),mask);
            % Computing AW and RW with their exact definitions an all K maps
            Y2_STF = RICEtools.get_STF_basis(2,CSphase,ComplexSTF,0);
            Y00 = Y2_STF(:,:,1);
            Y2m2 = Y2_STF(:,:,2);
            Y2m1 = Y2_STF(:,:,3);
            Y20 = Y2_STF(:,:,4);
            Y21 = Y2_STF(:,:,5);
            Y22 = Y2_STF(:,:,6);
            e1=0*dt_stf(1:3, :);
            parfor ii = 1:size(dt_stf,2)
                dt_current=dt_stf(:,ii);
                Dlm=dt_current(1:6);
                DT = Dlm(1)*Y00 + Dlm(2)*Y2m2 + Dlm(3)*Y2m1 + Dlm(4)*Y20 + Dlm(5)*Y21 + Dlm(6)*Y22;
                [eigvec, ~] = eigs(DT);
                e1(:, ii) = eigvec(:, 1);
            end
            e1 = RICEtools.vectorize(e1,mask);
        end
        % =================================================================
        function v1 = get_v1_from_4D_S2m(S2m,mask,CSphase,ComplexSTF,id_vi_4th)
            % 
            [ x, y, z, ~] = size(S2m);
            if ~exist('mask','var') || isempty(mask)
                mask = true(x, y, z);
            end
            if ~exist('CSphase','var') || isempty(CSphase) || CSphase
                CSphase=1; % 1 means we use it (default)
            else
                CSphase=0; % 0 means we DO NOT use it
            end
            if ~exist('ComplexSTF','var') || isempty(ComplexSTF) || ~ComplexSTF
                ComplexSTF=0; % 0 means we use real STF basis (default)
            else
                ComplexSTF=1; % 1 means we use complex STF basis
            end
            if ~exist('id_vi_4th','var') || isempty(id_vi_4th)
                id_vi_4th = 1; % 1 means we use it (default)
            end
            S2m = RICEtools.vectorize(S2m,mask);
            if size(S2m,1) == 5
                % ell = 2;
                v1 = zeros(3,size(S2m,2));
                % add zeros for ell=0
                S2m = [zeros(1,size(S2m,2)) ; S2m];
                Scart = RICEtools.STF2CART(S2m,CSphase,ComplexSTF);
                for ii = 1:size(S2m,2)
                    Dcart_3x3 = RICEtools.unflattenTensor(Scart(:,ii));
                    [V,~] = eig(Dcart_3x3);
                    v1(:,ii) = V(:,3);
                end
                v1 = RICEtools.vectorize(v1,mask);
            elseif size(S2m,1) == 9
                S2m = [zeros(6,size(S2m,2)) ; S2m];
                Scart = RICEtools.STF2CART(S2m,CSphase,ComplexSTF);
                for ii = 1:size(S2m,2)
                    Sarray = RICEtools.unflattenTensor(Scart(:,ii));
                    s4 = RICEtools.MapRank4_to_6x6(Sarray,'FullySymmetric');
                    [V,D] = eig(s4);
                    
        %             % Pick largest eigenvalue of rank-4 tensor (same for any rotation)
        %             [~,id_max]=max(diag(D));
        %             id_vi_4th = id_max;
                    
                    % Get main eigentensor
                    vi=V(:,id_vi_4th)/norm(V(:,id_vi_4th));
                    E6=[vi(1) 1/sqrt(2)*vi(4)  1/sqrt(2)*vi(5) ;  1/sqrt(2)*vi(4) vi(2) 1/sqrt(2)*vi(6) ;  1/sqrt(2)*vi(5) 1/sqrt(2)*vi(6) vi(3)];
                    [Vi,Di] = eig(E6);
                    % Ensure dets are +1
                    Vi = Vi*det(Vi);
                    v1(:,ii) = Vi(:,1);
                end
            else
                v1 = nan;
                return
            end
        end
        % =================================================================
        function [s, mask] = vectorize(S, mask)
            if nargin == 1
               mask = ~isnan(S(:,:,:,1));
            end
            if ismatrix(S)
                n = size(S, 1);
                [x, y, z] = size(mask);
                s = NaN([x, y, z, n], 'like', S);
                for i = 1:n
                    tmp = NaN(x, y, z, 'like', S);
                    tmp(mask(:)) = S(i, :);
                    s(:,:,:,i) = tmp;
                end
            else
                for i = 1:size(S, 4)
                    Si = S(:,:,:,i);
                    s(i, :) = Si(mask(:));
                end
            end
        end
        % =================================================================
        function dirs = get256dirs()
        % get 256 isotropically distributed directions
        dirs =  [0         0    1.0000;
            0.5924         0    0.8056;
            -0.7191   -0.1575   -0.6768;
            -0.9151   -0.3479    0.2040;
            0.5535    0.2437    0.7964;
            -0.0844    0.9609   -0.2636;
            0.9512   -0.3015    0.0651;
            -0.4225    0.8984    0.1202;
            0.5916   -0.6396    0.4909;
            0.3172    0.8818   -0.3489;
            -0.1988   -0.6687    0.7164;
            -0.2735    0.3047   -0.9123;
            0.9714   -0.1171    0.2066;
            -0.5215   -0.4013    0.7530;
            -0.3978   -0.9131   -0.0897;
            0.2680    0.8196    0.5063;
            -0.6824   -0.6532   -0.3281;
            0.4748   -0.7261   -0.4973;
            0.4504   -0.4036    0.7964;
            -0.5551   -0.8034   -0.2153;
            0.0455   -0.2169    0.9751;
            0.0483    0.5845    0.8099;
            -0.1909   -0.1544   -0.9694;
            0.8383    0.5084    0.1969;
            -0.2464    0.1148    0.9623;
            -0.7458    0.6318    0.2114;
            -0.0080   -0.9831   -0.1828;
            -0.2630    0.5386   -0.8005;
            -0.0507    0.6425   -0.7646;
            0.4476   -0.8877    0.1081;
            -0.5627    0.7710    0.2982;
            -0.3790    0.7774   -0.5020;
            -0.6217    0.4586   -0.6350;
            -0.1506    0.8688   -0.4718;
            -0.4579    0.2131    0.8631;
            -0.8349   -0.2124    0.5077;
            0.7682   -0.1732   -0.6163;
            0.0997   -0.7168   -0.6901;
            0.0386   -0.2146   -0.9759;
            0.9312    0.1655   -0.3249;
            0.9151    0.3053    0.2634;
            0.8081    0.5289   -0.2593;
            -0.3632   -0.9225    0.1305;
            0.2709   -0.3327   -0.9033;
            -0.1942   -0.9790   -0.0623;
            0.6302   -0.7641    0.1377;
            -0.6948   -0.3137    0.6471;
            -0.6596   -0.6452    0.3854;
            -0.9454    0.2713    0.1805;
            -0.2586   -0.7957    0.5477;
            -0.3576    0.6511    0.6695;
            -0.8490   -0.5275    0.0328;
            0.3830    0.2499   -0.8893;
            0.8804   -0.2392   -0.4095;
            0.4321   -0.4475   -0.7829;
            -0.5821   -0.1656    0.7961;
            0.3963    0.6637    0.6344;
            -0.7222   -0.6855   -0.0929;
            0.2130   -0.9650   -0.1527;
            0.4737    0.7367   -0.4825;
            -0.9956    0.0891    0.0278;
            -0.5178    0.7899   -0.3287;
            -0.8906    0.1431   -0.4317;
            0.2431   -0.9670    0.0764;
            -0.6812   -0.3807   -0.6254;
            -0.1091   -0.5141    0.8507;
            -0.2206    0.7274   -0.6498;
            0.8359    0.2674    0.4794;
            0.9873    0.1103    0.1147;
            0.7471    0.0659   -0.6615;
            0.6119   -0.2508    0.7502;
            -0.6191    0.0776    0.7815;
            0.7663   -0.4739    0.4339;
            -0.5699    0.5369    0.6220;
            0.0232   -0.9989    0.0401;
            0.0671   -0.4207   -0.9047;
            -0.2145    0.5538    0.8045;
            0.8554   -0.4894    0.1698;
            -0.7912   -0.4194    0.4450;
            -0.2341    0.0754   -0.9693;
            -0.7725    0.6346   -0.0216;
            0.0228    0.7946   -0.6067;
            0.7461   -0.3966   -0.5348;
            -0.4045   -0.0837   -0.9107;
            -0.4364    0.6084   -0.6629;
            0.6177   -0.3175   -0.7195;
            -0.4301   -0.0198    0.9026;
            -0.1489   -0.9706    0.1892;
            0.0879    0.9070   -0.4117;
            -0.7764   -0.4707   -0.4190;
            0.9850    0.1352   -0.1073;
            -0.1581   -0.3154    0.9357;
            0.8938   -0.3246    0.3096;
            0.8358   -0.4464   -0.3197;
            0.4943    0.4679    0.7327;
            -0.3095    0.9015   -0.3024;
            -0.3363   -0.8942   -0.2956;
            -0.1271   -0.9274   -0.3519;
            0.3523   -0.8717   -0.3407;
            0.7188   -0.6321    0.2895;
            -0.7447    0.0924   -0.6610;
            0.1622    0.7186    0.6762;
            -0.9406   -0.0829   -0.3293;
            -0.1229    0.9204    0.3712;
            -0.8802    0.4668    0.0856;
            -0.2062   -0.1035    0.9730;
            -0.4861   -0.7586   -0.4338;
            -0.6138    0.7851    0.0827;
            0.8476    0.0504    0.5282;
            0.3236    0.4698   -0.8213;
            -0.7053   -0.6935    0.1473;
            0.1511    0.3778    0.9135;
            0.6011    0.5847    0.5448;
            0.3610    0.3183    0.8766;
            0.9432    0.3304    0.0341;
            0.2423   -0.8079   -0.5372;
            0.4431   -0.1578    0.8825;
            0.6204    0.5320   -0.5763;
            -0.2806   -0.5376   -0.7952;
            -0.5279   -0.8071    0.2646;
            -0.4214   -0.6159    0.6656;
            0.6759   -0.5995   -0.4288;
            0.5670    0.8232   -0.0295;
            -0.0874    0.4284   -0.8994;
            0.8780   -0.0192   -0.4782;
            0.0166    0.8421    0.5391;
            -0.7741    0.2931   -0.5610;
            0.9636   -0.0579   -0.2611;
            0         0   -1.0000;
            -0.5924         0   -0.8056;
            0.7191    0.1575    0.6768;
            0.9151    0.3479   -0.2040;
            -0.5535   -0.2437   -0.7964;
            0.0844   -0.9609    0.2636;
            -0.9512    0.3015   -0.0651;
            0.4225   -0.8984   -0.1202;
            -0.5916    0.6396   -0.4909;
            -0.3172   -0.8818    0.3489;
            0.1988    0.6687   -0.7164;
            0.2735   -0.3047    0.9123;
            -0.9714    0.1171   -0.2066;
            0.5215    0.4013   -0.7530;
            0.3978    0.9131    0.0897;
            -0.2680   -0.8196   -0.5063;
            0.6824    0.6532    0.3281;
            -0.4748    0.7261    0.4973;
            -0.4504    0.4036   -0.7964;
            0.5551    0.8034    0.2153;
            -0.0455    0.2169   -0.9751;
            -0.0483   -0.5845   -0.8099;
            0.1909    0.1544    0.9694;
            -0.8383   -0.5084   -0.1969;
            0.2464   -0.1148   -0.9623;
            0.7458   -0.6318   -0.2114;
            0.0080    0.9831    0.1828;
            0.2630   -0.5386    0.8005;
            0.0507   -0.6425    0.7646;
            -0.4476    0.8877   -0.1081;
            0.5627   -0.7710   -0.2982;
            0.3790   -0.7774    0.5020;
            0.6217   -0.4586    0.6350;
            0.1506   -0.8688    0.4718;
            0.4579   -0.2131   -0.8631;
            0.8349    0.2124   -0.5077;
            -0.7682    0.1732    0.6163;
            -0.0997    0.7168    0.6901;
            -0.0386    0.2146    0.9759;
            -0.9312   -0.1655    0.3249;
            -0.9151   -0.3053   -0.2634;
            -0.8081   -0.5289    0.2593;
            0.3632    0.9225   -0.1305;
            -0.2709    0.3327    0.9033;
            0.1942    0.9790    0.0623;
            -0.6302    0.7641   -0.1377;
            0.6948    0.3137   -0.6471;
            0.6596    0.6452   -0.3854;
            0.9454   -0.2713   -0.1805;
            0.2586    0.7957   -0.5477;
            0.3576   -0.6511   -0.6695;
            0.8490    0.5275   -0.0328;
            -0.3830   -0.2499    0.8893;
            -0.8804    0.2392    0.4095;
            -0.4321    0.4475    0.7829;
            0.5821    0.1656   -0.7961;
            -0.3963   -0.6637   -0.6344;
            0.7222    0.6855    0.0929;
            -0.2130    0.9650    0.1527;
            -0.4737   -0.7367    0.4825;
            0.9956   -0.0891   -0.0278;
            0.5178   -0.7899    0.3287;
            0.8906   -0.1431    0.4317;
            -0.2431    0.9670   -0.0764;
            0.6812    0.3807    0.6254;
            0.1091    0.5141   -0.8507;
            0.2206   -0.7274    0.6498;
            -0.8359   -0.2674   -0.4794;
            -0.9873   -0.1103   -0.1147;
            -0.7471   -0.0659    0.6615;
            -0.6119    0.2508   -0.7502;
            0.6191   -0.0776   -0.7815;
            -0.7663    0.4739   -0.4339;
            0.5699   -0.5369   -0.6220;
            -0.0232    0.9989   -0.0401;
            -0.0671    0.4207    0.9047;
            0.2145   -0.5538   -0.8045;
            -0.8554    0.4894   -0.1698;
            0.7912    0.4194   -0.4450;
            0.2341   -0.0754    0.9693;
            0.7725   -0.6346    0.0216;
            -0.0228   -0.7946    0.6067;
            -0.7461    0.3966    0.5348;
            0.4045    0.0837    0.9107;
            0.4364   -0.6084    0.6629;
            -0.6177    0.3175    0.7195;
            0.4301    0.0198   -0.9026;
            0.1489    0.9706   -0.1892;
            -0.0879   -0.9070    0.4117;
            0.7764    0.4707    0.4190;
            -0.9850   -0.1352    0.1073;
            0.1581    0.3154   -0.9357;
            -0.8938    0.3246   -0.3096;
            -0.8358    0.4464    0.3197;
            -0.4943   -0.4679   -0.7327;
            0.3095   -0.9015    0.3024;
            0.3363    0.8942    0.2956;
            0.1271    0.9274    0.3519;
            -0.3523    0.8717    0.3407;
            -0.7188    0.6321   -0.2895;
            0.7447   -0.0924    0.6610;
            -0.1622   -0.7186   -0.6762;
            0.9406    0.0829    0.3293;
            0.1229   -0.9204   -0.3712;
            0.8802   -0.4668   -0.0856;
            0.2062    0.1035   -0.9730;
            0.4861    0.7586    0.4338;
            0.6138   -0.7851   -0.0827;
            -0.8476   -0.0504   -0.5282;
            -0.3236   -0.4698    0.8213;
            0.7053    0.6935   -0.1473;
            -0.1511   -0.3778   -0.9135;
            -0.6011   -0.5847   -0.5448;
            -0.3610   -0.3183   -0.8766;
            -0.9432   -0.3304   -0.0341;
            -0.2423    0.8079    0.5372;
            -0.4431    0.1578   -0.8825;
            -0.6204   -0.5320    0.5763;
            0.2806    0.5376    0.7952;
            0.5279    0.8071   -0.2646;
            0.4214    0.6159   -0.6656;
            -0.6759    0.5995    0.4288;
            -0.5670   -0.8232    0.0295;
            0.0874   -0.4284    0.8994;
            -0.8780    0.0192    0.4782;
            -0.0166   -0.8421   -0.5391;
            0.7741   -0.2931    0.5610;
            -0.9636    0.0579    0.2611];
        end
        % =================================================================
        function dirs = radialsampling(dir, n)        
            % compute Equator Points
            dt = 2*pi/n;
            theta = 0:dt:(2*pi-dt);
            dirs = [cos(theta)', sin(theta)', 0*theta']';
            v = [-dir(2), dir(1), 0];
            s = sqrt(sum(v.^2));
            c = dir(3);
            V = [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
            R = eye(3) + V + V*V * (1-c)/s^2;
            dirs = R*dirs;
        end
        % =================================================================
        function WrapperPlotManySlices(ARRAY_4D, slice,clims,names,Nrows,positions,nanTransparent,colorbar_flag)
            % Plot many slices
            sz=size(ARRAY_4D);
            if length(sz)==4
                Nplots=sz(4);
            elseif length(sz)==3
                Nplots=sz(3);
            end
            
            if ~exist('positions', 'var') || isempty(positions)
                positions=1:Nplots;
            end
            if ~exist('nanTransparent', 'var') || isempty(nanTransparent)
                nanTransparent=0;
            end
            if ~exist('colorbar_flag', 'var') || isempty(colorbar_flag)
                colorbar_flag=1;
            end
            
            if isvector(clims)
                clims=repmat(clims,Nplots,1);
            end
            
            for ii=1:Nplots
                subplot(Nrows,ceil(Nplots/Nrows),positions(ii))
                
                if ~nanTransparent
                    if length(sz)==4
                        imagesc(ARRAY_4D(:,:,slice,ii),clims(ii,:))
                    elseif length(sz)==3
                        imagesc(ARRAY_4D(:,:,ii),clims(ii,:))
                    end
                else
                    if length(sz)==4
                        h=pcolor(ARRAY_4D(:,:,slice,ii)); clim(clims(ii,:)),set(h, 'EdgeColor', 'none');
                    elseif length(sz)==3
                        h=pcolor(ARRAY_4D(:,:,ii)); clim(clims(ii,:)),set(h, 'EdgeColor', 'none');
                    end
                end
                
                if isempty(names)
                    title(['case ',num2str(ii)],'interpreter','latex')
                else
                    title(names{ii},'interpreter','latex')
                end
                set(gca,'FontSize',30), axis off, grid off, axis equal
                
                if colorbar_flag
                cb=colorbar('south'); 
                cb_pos =  cb.Position; cb_pos(2)=cb_pos(2)-0.025; bias=0.03; cb_pos(1)=cb_pos(1)+bias; cb_pos(3)=cb_pos(3)-2*bias;
                set(cb,'position',cb_pos)
                cb.Ticks=clims(ii,:);    
                cb.Color= [0 0 0];
                cb.Label.HorizontalAlignment='left';
            %     cb.Label.HorizontalAlignment='center';
                end
            end
        end
        % =================================================================
        function y = getY(Lmax) % updated Sept2024 for Racah normalization
            % Builds y{L/2,1+l/2,1+l+m}, L=2,4,6...,Lmax, l=2,4,...,L, m=-l...l, 
            % symmetric matrices to generate real spherical harmonics
            %
            % (c) Dmitry S. Novikov, November 2014; dima@alum.mit.edu
            
            ylm = RICEtools.getYreal(Lmax);
            
            % L=2, l=0
            y{1,1,1} = eye(3); 
            
            % L=4,...,Lmax, l=0
            for L=4:2:Lmax 
                y{L/2,1,1} = kron(y{(L-2)/2,1,1}, eye(3)); %/sqrt(4*pi); 
            end
            
            % L=2,...,Lmax, l=L
            for L=2:2:Lmax
                for m=-L:L
                    y{L/2,1+L/2,1+L+m} = ylm{L/2,1+L+m}; 
                end
            end
            
            % L=4,...,Lmax, l=2,...,L-2
            for L=4:2:Lmax
                shape=3*(ones(1,L));
                for l=2:2:L-2
                    for m=-l:l
                        y{L/2,1+l/2,1+l+m} = RICEtools.symmetrizeTensor(reshape(kron(reshape(ylm{l/2,1+l+m}, [3^(l/2),3^(l/2)]), y{(L-l)/2,1,1}), shape)); 
                    end
                end
            end
            
            for L=2:2:Lmax 
                y{L/2,1,1} = RICEtools.symmetrizeTensor(reshape(y{L/2,1,1}, 3*ones(1,L)));%/sqrt(4*pi); 
            end
        end
        % =================================================================
        function ylmreal = getYreal(L) % updated Sept2024 for Racah normalization
        % Builds y{l,1+l+m}, l=1,2,3,4...,L, m=-l...l, REAL-valued 
        % location-independent STF tensors to generate tesseral spherical harmonics
        % according to Eq (2.12) of KS Thorne, Rev Mod Phys 52, 299 (1980)
        %
        % (c) Dmitry S. Novikov, November 2014; dima@alum.mit.edu
            ylm = RICEtools.getYcomplex(L);
            for l = 2:2:L 
                ylmreal{l/2,1+l} = ylm{l/2,1+l};
                for m=1:l
                    ylmreal{l/2,1+l+m} = sqrt(2) * (-1)^0 * real(ylm{l/2,1+l+m}); % m > 0 ; was (-1)^m 
                    ylmreal{l/2,1+l-m} = sqrt(2) * (-1)^0 * imag(ylm{l/2,1+l+m}); % m < 0 ; was (-1)^m
                   
                    %ylmreal{l/2,1+l+m} = (ylm{l/2,1+l-m} + (-1)^m * ylm{l/2,1+l+m})/sqrt(2); 
                    %ylmreal{l/2,1+l-m} = (-1)^(m+1) * 1i*(ylm{l/2,1+l+m} - (-1)^m * ylm{l/2,1+l-m})/sqrt(2); % extra (-1)^(m+1) 
                    %ylmreal{l/2,1+l-m} = 1i*(ylm{l/2,1+l+m} - (-1)^m * ylm{l/2,1+l-m})/sqrt(2);  % Thorne
              
                    %ylmreal{l/2,1+l+m} = imag(ylm{l/2,1+l+m})*sqrt(2); % Thorne
                    %ylmreal{l/2,1+l-m} = imag(ylm{l/2,1+l-m})*sqrt(2); % Thorne
                    
                end
            end
        end
        % =================================================================
        function ylm = getYcomplex(L) % updated Sept2024 for Racah normalization
            % Builds y{l,1+l+m}, l=1,2,3,4...,L, m=-l...l, 
            % location-independent STF tensors to generate complex spherical harmonics
            % according to Eq (2.12) of KS Thorne, Rev Mod Phys 52, 299 (1980)
            %
            % (c) Dmitry S. Novikov, November 2014; dima@alum.mit.edu
            x=[1;2;3]; x1=(x==1); x2=(x==2); x3=(x==3);
            Clmj = @(l,m,j) (-1)^(m+j) ...
                        * sqrt(factorial(l-m)/factorial(l+m)) * factorial(2*(l-j)) ... 
                        / (2^l * factorial(j) * factorial(l-j) * factorial(l-m-2*j));
            for l=2:2:L
                shape=3*ones(1,l);
                for m=0:l
                    ylm{l/2,1+l+m}=reshape(zeros(1,3^l),shape);
                    for j=0:floor((l-m)/2)
                        X=1;
                        for n=1:m X=kron(X, x1+1i*x2); end
                        for n=m+1:l-2*j X=kron(X,x3); end
                        for n=l-2*j+1:2:l X=kron(X,eye(3)); end
                        ylm{l/2,1+l+m} = ylm{l/2,1+l+m} + Clmj(l,m,j)*reshape(X,shape);
                    end
                    ylm{l/2,1+l+m} = RICEtools.symmetrizeTensor(ylm{l/2,1+l+m});
                    ylm{l/2,1+l-m} = (-1)^m * conj(ylm{l/2,1+l+m});
                end
            end
        end     
        % =================================================================
        function Slm = CART2STF(Scart,CSphase,ComplexSTF) % updated Sept2024 for Racah normalization
            % Scart is a 2D array containing
            % [ 6 x M] [S_11;S_22;S_33;S_12;S_13;S_23] 
            % or 
            % [ 15 x M]: [S_1111;S_2222;S_3333;S_1122;S_1133;S_1112;S_1113;S_1123;S_2233;S_2212;S_2213;S_2223;S_3312;S_3313;S_3323];   
            %
            % Slm is a 2D array containing
            % [ 6 x M]  [S_00;S_2-2;S_2-1;S_20;S_21;S_22]
            % or 
            % [ 15 x M]: [S_00;S_2-2;S_2-1;S_20;S_21;S_22;S_4-4;S_4-3;S_4-2;S_4-1;S_40;S_41;S_42;S_43;S_44]; 
            if ~exist('CSphase','var') || isempty(CSphase) || CSphase
                CSphase=1; % 1 means we use it (default)
            else
                CSphase=0; % 0 means we DO NOT use it
            end
            if ~exist('ComplexSTF','var') || isempty(ComplexSTF) || ~ComplexSTF
                ComplexSTF=0; % 0 means we use real STF basis (default)
            else
                ComplexSTF=1; % 1 means we use complex STF basis
            end
%             C0=sqrt(1/(4*pi)); C2=sqrt(5/(4*pi)); C4=sqrt(9/(4*pi));
            % Fixing C_ell to 1 for Racah's normalization
            C0 = 1; C2 = 1; C4 = 1;
            if size(Scart,1)==6
                Y2 = RICEtools.get_STF_basis(2,CSphase,ComplexSTF);
                S_cnt=[1 1 1 2 2 2];
                id_indep_elemens=[1 5 9 4 7 8];
                S00 = 1/3*(1/C0^2)*(Y2(id_indep_elemens,1)'.*S_cnt)*Scart;
%                 S00 = (Y2(id_indep_elemens,1)'.*S_cnt)*Scart;
                S2m = 2/3*(1/C2^2)*(Y2(id_indep_elemens,2:6)'.*S_cnt)*Scart;
                Slm = [S00; S2m];
            elseif size(Scart,1)==15
                Y4 = RICEtools.get_STF_basis(4,CSphase,ComplexSTF);
                S_cnt=[1 1 1 6 6 4 4 12 6 4 12 4 12 4 4];
                id_indep_elemens=[1 41 81 37 73 28 55 64 77 32 59 68 36 63 72];
                S00 = 1/5*(1/C0^2)*(Y4(id_indep_elemens,1)'.*S_cnt)*Scart;
%                 S00 = (Y4(id_indep_elemens,1)'.*S_cnt)*Scart;
                S2m = 4/7*(1/C2^2)*(Y4(id_indep_elemens,2:6)'.*S_cnt)*Scart;
                S4m = 8/35*(1/C4^2)*(Y4(id_indep_elemens,7:15)'.*S_cnt)*Scart;
                Slm = [S00; S2m; S4m];
            else
                error('Only rank-2 and rank-4 fully symmetric tensors are supported')
            end
        end         
        % =================================================================
        function Scart = STF2CART(Slm,CSphase,ComplexSTF) % updated Sept2024 for Racah normalization
            % Scart is a 2D array containing
            % [ 6 x M] [S_11;S_22;S_33;S_12;S_13;S_23] 
            % or 
            % [ 15 x M]: [S_1111;S_2222;S_3333;S_1122;S_1133;S_1112;S_1113;S_1123;S_2233;S_2212;S_2213;S_2223;S_3312;S_3313;S_3323];   
            %
            % Slm is a 2D array containing
            % [ 6 x M]  [S_00;S_2-2;S_2-1;S_20;S_21;S_22]
            % or 
            % [ 15 x M]: [S_00;S_2-2;S_2-1;S_20;S_21;S_22;S_4-4;S_4-3;S_4-2;S_4-1;S_40;S_41;S_42;S_43;S_44]; 
            if ~exist('CSphase','var') || isempty(CSphase) || CSphase
                CSphase=1; % 1 means we use it (default)
            else
                CSphase=0; % 0 means we DO NOT use it
            end
            if ~exist('ComplexSTF','var') || isempty(ComplexSTF) || ~ComplexSTF
                ComplexSTF=0; % 0 means we use real STF basis (default)
            else
                ComplexSTF=1; % 1 means we use complex STF basis
            end
            if size(Slm,1)==6
                Y2 = RICEtools.get_STF_basis(2,CSphase,ComplexSTF);
                Scart = Y2*Slm;
                keep = [1 5 9 2 3 6];
                Scart = Scart(keep,:);
            elseif size(Slm,1)==15
                Y4 = RICEtools.get_STF_basis(4,CSphase,ComplexSTF);
                Scart = Y4*Slm;
                keep = [1 41 81 37 73 28 55 64 77 32 59 68 36 63 72];
                Scart = Scart(keep,:);
            else
                error('Only rank-2 and rank-4 fully symmetric tensors are supported')
            end
        end        
        % =================================================================
        function S = BuildSTF(Slm,L,CSphase,ComplexSTF) % updated Sept2024 for Racah normalization
            % S = BuildSTF(Slm,L,CSphase,ComplexSTF)
            %
            % Building fully symmetric tensor from STF parametrization
            %
            % By: Santiago Coelho (01/02/2023)
            if ~isvector(Slm)
                error('Slm input should be a vector')
            end
            Nlm = length(Slm);
            if ~exist('L', 'var') || isempty(L)
                if Nlm>6
                    L = 4;
                else
                    L = 2;
                end
            end
            if ~exist('CSphase','var') || isempty(CSphase) || CSphase
                CSphase=1; % 1 means we use it (default)
            else
                CSphase=0; % 0 means we DO NOT use it
            end
            if ~exist('ComplexSTF','var') || isempty(ComplexSTF) || ~ComplexSTF
                ComplexSTF=0; % 0 means we use real STF basis (default)
            else
                ComplexSTF=1; % 1 means we use complex STF basis
            end
            if L==2
                Y_ell = RICEtools.get_STF_basis(L,CSphase,ComplexSTF,0) ;
                if Nlm==1
                    S=Slm(1)*Y_ell(:,:,1);
                elseif Nlm==5
                    S=Slm(1)*Y_ell(:,:,2)+Slm(2)*Y_ell(:,:,3)+Slm(3)*Y_ell(:,:,4)+Slm(4)*Y_ell(:,:,5)+Slm(5)*Y_ell(:,:,6);
                elseif Nlm==6
                    S=Slm(1)*Y_ell(:,:,1)+Slm(2)*Y_ell(:,:,2)+Slm(3)*Y_ell(:,:,3)+Slm(4)*Y_ell(:,:,4)+Slm(5)*Y_ell(:,:,5)+Slm(6)*Y_ell(:,:,6);
                end
            elseif L==4
                Y_ell = RICEtools.get_STF_basis(L,CSphase,ComplexSTF,0) ;
                if Nlm==1
                    S=Slm(1)*Y_ell(:,:,:,:,1);
                elseif Nlm==5
                    S=Slm(1)*Y_ell(:,:,:,:,2)+Slm(2)*Y_ell(:,:,:,:,3)+Slm(3)*Y_ell(:,:,:,:,4)+Slm(4)*Y_ell(:,:,:,:,5)+Slm(5)*Y_ell(:,:,:,:,6);
                elseif Nlm==6
                    S=Slm(1)*Y_ell(:,:,:,:,1)+Slm(2)*Y_ell(:,:,:,:,2)+Slm(3)*Y_ell(:,:,:,:,3)+Slm(4)*Y_ell(:,:,:,:,4)+Slm(5)*Y_ell(:,:,:,:,5)+Slm(6)*Y_ell(:,:,:,:,6);
                elseif Nlm==9
                    S=Slm(1)*Y_ell(:,:,:,:,7)+Slm(2)*Y_ell(:,:,:,:,8)+Slm(3)*Y_ell(:,:,:,:,9)+Slm(4)*Y_ell(:,:,:,:,10)+Slm(5)*Y_ell(:,:,:,:,11)+Slm(6)*Y_ell(:,:,:,:,12)+Slm(7)*Y_ell(:,:,:,:,13)+Slm(8)*Y_ell(:,:,:,:,14)+Slm(9)*Y_ell(:,:,:,:,15);
                elseif Nlm==15
                    S=Slm(1)*Y_ell(:,:,:,:,1)+Slm(2)*Y_ell(:,:,:,:,2)+Slm(3)*Y_ell(:,:,:,:,3)+Slm(4)*Y_ell(:,:,:,:,4)+Slm(5)*Y_ell(:,:,:,:,5)+Slm(6)*Y_ell(:,:,:,:,6)+...
                      Slm(7)*Y_ell(:,:,:,:,7)+Slm(8)*Y_ell(:,:,:,:,8)+Slm(9)*Y_ell(:,:,:,:,9)+Slm(10)*Y_ell(:,:,:,:,10)+Slm(11)*Y_ell(:,:,:,:,11)+Slm(12)*Y_ell(:,:,:,:,12)+Slm(13)*Y_ell(:,:,:,:,13)+Slm(14)*Y_ell(:,:,:,:,14)+Slm(15)*Y_ell(:,:,:,:,15);
                end
            end
        end
        % =================================================================
        function Sflat = flattenTensor(Sarray) % updated Sept2024 for Racah normalization
            % IMPORTANT: this function assumes this is a cartesian tensor (rank-2 or rank-4)
            % inputs are 2D or 4D arrays
            % outputs will be:
            % [ 6 x 1] [S_11;S_22;S_33;S_12;S_13;S_23] 
            % or 
            % [ 15 x 1]: [S_1111;S_2222;S_3333;S_1122;S_1133;S_1112;S_1113;S_1123;S_2233;S_2212;S_2213;S_2223;S_3312;S_3313;S_3323];   
            dim = length(size(Sarray));
            if dim == 2
                Sflat = [Sarray(1,1) ; Sarray(2,2) ; Sarray(3,3) ; Sarray(1,2) ; Sarray(1,3) ; Sarray(2,3)];
            elseif dim == 4
                S1111 = Sarray(1,1,1,1);
                S1112 = Sarray(1,1,1,2);
                S1113 = Sarray(1,1,1,3);
                S1122 = Sarray(1,1,2,2);
                S1123 = Sarray(1,1,2,3);
                S1133 = Sarray(1,1,3,3);
                S1222 = Sarray(1,2,2,2); S2212 = S1222;
                S1223 = Sarray(1,2,2,3); S2213 = S1223;
                S1233 = Sarray(1,2,3,3); S3312 = S1233;
                S1333 = Sarray(1,3,3,3); S3313 = S1333;
                S2222 = Sarray(2,2,2,2);
                S2223 = Sarray(2,2,2,3);
                S2233 = Sarray(2,2,3,3);
                S2333 = Sarray(2,3,3,3); S3323 = S2333;
                S3333 = Sarray(3,3,3,3);
                Sflat = [S1111 ; S2222 ; S3333 ; S1122 ; S1133 ; S1112 ; S1113 ; S1123 ; S2233 ; S2212 ; S2213 ; S2223 ; S3312 ; S3313 ; S3323] ;
            else
                error('Only dim = 2 (Lmax = 2) or dim = 4 (Lmax = 4) are supported')
            end
        end
        % =================================================================
        function Sarray = unflattenTensor(Sflat) % updated Sept2024 for Racah normalization
            % IMPORTANT: this function assumes this is a cartesian tensor (rank-2 or rank-4)
            % inputs will be:
            % [ 6 x 1] [S_11;S_22;S_33;S_12;S_13;S_23] 
            % or 
            % [ 15 x 1]: [S_1111;S_2222;S_3333;S_1122;S_1133;S_1112;S_1113;S_1123;S_2233;S_2212;S_2213;S_2223;S_3312;S_3313;S_3323];   
            % outputs will be 2D or 4D arrays
            if ~isvector(Sflat)
                error('Sflat input should be a vector')
            end            
            len = length(Sflat);
            if len == 6
                Sarray = [Sflat(1), Sflat(4), Sflat(5) ; Sflat(4), Sflat(2), Sflat(6) ; Sflat(5), Sflat(6), Sflat(3)];
            elseif len == 15
                % C0=sqrt(1/(4*pi)); C2=sqrt(5/(4*pi)); C4=sqrt(9/(4*pi));
                % Fixing Racah's normalization
                CSphase = 1; ComplexSTF = 1;
                Y4 = RICEtools.get_STF_basis(4,CSphase,ComplexSTF);
                S_cnt=[1 1 1 6 6 4 4 12 6 4 12 4 12 4 4];
                id_indep_elemens=[1 41 81 37 73 28 55 64 77 32 59 68 36 63 72];
                S00 = 1/5*(Y4(id_indep_elemens,1)'.*S_cnt)*Sflat(:);
                S2m = 4/7*(Y4(id_indep_elemens,2:6)'.*S_cnt)*Sflat(:);
                S4m = 8/35*(Y4(id_indep_elemens,7:15)'.*S_cnt)*Sflat(:);
                Slm = [S00; S2m; S4m];
                Sarray = RICEtools.BuildSTF(Slm,4,CSphase,ComplexSTF);
            else
                error('Only length = 6 (rank = 2) or length = 15 (rank = 4) are supported')
            end
        end
        % =================================================================
        function [A_6x6, A_21D] = MapRank4_to_6x6(Input,type) % updated Sept2024 for Racah normalization
            % [A_6x6, A_21D] = MapRank4_to_6x6(Input,type)
            % 
            % This function maps a rank-4 tensor (in 3D) to a 6x6 matrix
            %
            % - if type = 'FullySymmetric' the tensor is assumed to be fully symmetric (15 degrees of freedom)
            % - if type = 'MinorMajor' the tensor is assumed to have minor and major symmetries (21 degrees of freedom)
            %
            % Santiago Coelho 25/01/2023
            if strcmp(type,'MinorMajor')
                C1111 = Input(1,1,1,1);
                C1112 = Input(1,1,1,2);
                C1113 = Input(1,1,1,3);
                C1122 = Input(1,1,2,2);
                C1123 = Input(1,1,2,3);
                C1133 = Input(1,1,3,3);
                C1222 = Input(1,2,2,2);
                C1223 = Input(1,2,2,3);
                C1233 = Input(1,2,3,3);
                C1333 = Input(1,3,3,3);
                C2222 = Input(2,2,2,2);
                C2223 = Input(2,2,2,3);
                C2233 = Input(2,2,3,3);
                C2333 = Input(2,3,3,3);
                C3333 = Input(3,3,3,3);
                C1322 = Input(1,3,2,2);
                C2323 = Input(2,3,2,3);
                C1313 = Input(1,3,1,3);
                C1212 = Input(1,2,1,2);
                C1323 = Input(1,3,2,3);
                C1213 = Input(1,2,1,3);
                C2212 = C1222; C3312 = C1233; C3313 = C1333; C3323 = C2333; C2213 = C1322;
                A_6x6 = [ C1111         , C1122         , C1133         , sqrt(2)*C1112 , sqrt(2)*C1113 , sqrt(2)*C1123 ; ...
                          C1122         , C2222         , C2233         , sqrt(2)*C1222 , sqrt(2)*C1322 , sqrt(2)*C2223 ; ...
                          C1133         , C2233         , C3333         , sqrt(2)*C1233 , sqrt(2)*C1333 , sqrt(2)*C2333 ; ...
                          sqrt(2)*C1112 , sqrt(2)*C1222 , sqrt(2)*C1233 , 2*C1212       , 2*C1213       , 2*C1223       ; ...
                          sqrt(2)*C1113 , sqrt(2)*C1322 , sqrt(2)*C1333 , 2*C1213       , 2*C1313       , 2*C1323       ; ...
                          sqrt(2)*C1123 , sqrt(2)*C2223 , sqrt(2)*C2333 , 2*C1223       , 2*C1323       , 2*C2323       ];
                A_21D = [C1111;C2222;C3333;C1212;C1313;C2323;C1122;C1133;C1112;C1113;C1123;C2233;C2212;C2213;C2223;C3312;C3313;C3323;C1213;C1223;C1323];
            elseif strcmp(type,'FullySymmetric')
                S1111=Input(1,1,1,1);
                S1112=Input(1,1,1,2);
                S1113=Input(1,1,1,3);
                S1122=Input(1,1,2,2);
                S1123=Input(1,1,2,3);
                S1133=Input(1,1,3,3);
                S1222=Input(1,2,2,2);
                S1223=Input(1,2,2,3);
                S1233=Input(1,2,3,3);
                S1333=Input(1,3,3,3);
                S2222=Input(2,2,2,2);
                S2223=Input(2,2,2,3);
                S2233=Input(2,2,3,3);
                S2333=Input(2,3,3,3);
                S3333=Input(3,3,3,3);
                S2212 = S1222; S2213 = S1223; S3312 = S1233; S3313 = S1333; S3323 = S2333;
                % NEW
                A_6x6 = [ S1111         , S1122         , S1133         , sqrt(2)*S1112 , sqrt(2)*S1113 , sqrt(2)*S1123 ; ...
                          S1122         , S2222         , S2233         , sqrt(2)*S1222 , sqrt(2)*S1223 , sqrt(2)*S2223 ; ...
                          S1133         , S2233         , S3333         , sqrt(2)*S1233 , sqrt(2)*S1333 , sqrt(2)*S2333 ; ...
                          sqrt(2)*S1112 , sqrt(2)*S1222 , sqrt(2)*S1233 , 2*S1122       , 2*S1123       , 2*S1223 ; ...
                          sqrt(2)*S1113 , sqrt(2)*S1223 , sqrt(2)*S1333 , 2*S1123       , 2*S1133       , 2*S1233 ; ...
                          sqrt(2)*S1123 , sqrt(2)*S2223 , sqrt(2)*S2333 , 2*S1223       , 2*S1233       , 2*S2233 ];
                A_21D = [S1111;S2222;S3333;S1122;S1133;S1112;S1113;S1123;S2233;S2212;S2213;S2223;S3312;S3313;S3323];
            end        
        end
        % =================================================================
    function [S_cart_15,Csym_cart_21] = reshape_symmetrize_C_cartesian(C_cart_21) % updated Sept2024 for Racah normalization
            % [S_cart_15,Csym_cart_21] = reshape_symmetrize_C_cartesian(C_cart_21)
            %
            % C: 2D array containing [C_1111;C_2222;C_3333;C_1212;C_1313;C_2323;C_1122;...
            %                         C_1133;C_1112;C_1113;C_1123;C_2233;C_2212;C_2213;...
            %                         C_2223;C_3312;C_3313;C_3323;C_1213;C_1223;C_1323];
            % S: 2D array containing [S_1111;S_2222;S_3333;S_1122;S_1133;...
            %                         S_1112;S_1113;S_1123;S_2233;S_2212;...
            %                         S_2213;S_2223;S_3312;S_3313;S_3323];   
            % Csym:  2D array containing the repeated elements of S in the C's format
            sz = size(C_cart_21);
            if length(sz) == 4
                only_one_C_rank4 = 1;
                [~, C_cart_21_unique] = RICEtools.MapRank4_to_6x6(C_cart_21,'MinorMajor');
                C_cart_21 = repmat(C_cart_21_unique,1,5);
            else
                only_one_C_rank4 = 0;
            end
            S_cart_15=C_cart_21(1:15,:)*0;
            S_cart_15(1,:)=C_cart_21(1,:);
            S_cart_15(2,:)=C_cart_21(2,:);
            S_cart_15(3,:)=C_cart_21(3,:);
            S_cart_15(4,:)=1/3*C_cart_21(7,:)+2/3*C_cart_21(4,:);
            S_cart_15(5,:)=1/3*C_cart_21(8,:)+2/3*C_cart_21(5,:);
            S_cart_15(6,:)=C_cart_21(9,:);
            S_cart_15(7,:)=C_cart_21(10,:);
            S_cart_15(8,:)=1/3*C_cart_21(11,:)+2/3*C_cart_21(19,:);
            S_cart_15(9,:)=1/3*C_cart_21(12,:)+2/3*C_cart_21(6,:);
            S_cart_15(10,:)=C_cart_21(13,:);
            S_cart_15(11,:)=1/3*C_cart_21(14,:)+2/3*C_cart_21(20,:);
            S_cart_15(12,:)=C_cart_21(15,:);
            S_cart_15(13,:)=1/3*C_cart_21(16,:)+2/3*C_cart_21(21,:);
            S_cart_15(14,:)=C_cart_21(17,:);
            S_cart_15(15,:)=C_cart_21(18,:);
            Csym_cart_21=C_cart_21*0;
            Csym_cart_21(1,:)=S_cart_15(1,:);
            Csym_cart_21(2,:)=S_cart_15(2,:);
            Csym_cart_21(3,:)=S_cart_15(3,:);
            Csym_cart_21(4,:)=S_cart_15(4,:);%4-
            Csym_cart_21(5,:)=S_cart_15(5,:);%5-
            Csym_cart_21(6,:)=S_cart_15(9,:);%9-
            Csym_cart_21(7,:)=S_cart_15(4,:);%4-
            Csym_cart_21(8,:)=S_cart_15(5,:);%5-
            Csym_cart_21(9,:)=S_cart_15(6,:);
            Csym_cart_21(10,:)=S_cart_15(7,:);
            Csym_cart_21(11,:)=S_cart_15(8,:);%8-
            Csym_cart_21(12,:)=S_cart_15(9,:);%9-
            Csym_cart_21(13,:)=S_cart_15(10,:);
            Csym_cart_21(14,:)=S_cart_15(11,:);%11-
            Csym_cart_21(15,:)=S_cart_15(12,:);
            Csym_cart_21(16,:)=S_cart_15(13,:);%13-
            Csym_cart_21(17,:)=S_cart_15(14,:);
            Csym_cart_21(18,:)=S_cart_15(15,:);
            Csym_cart_21(19,:)=S_cart_15(8,:);%8-
            Csym_cart_21(20,:)=S_cart_15(11,:);%11-
            Csym_cart_21(21,:)=S_cart_15(13,:);%13-
            if only_one_C_rank4
                S_cart_15 = S_cart_15(:,1);
                Csym_cart_21 = Csym_cart_21(:,1);
            end
        end
        % =================================================================
        function [S_cart_21] = reshape_S_from15to21_elements(S_cart_15) % updated Sept2024 for Racah normalization
            % [S_cart_21] = reshape_S_from15to21_elements(S_cart_15)
            %
            % S_cart_15: 2D array containing [S_1111;S_2222;S_3333;S_1122;S_1133;...
            %                                 S_1112;S_1113;S_1123;S_2233;S_2212;...
            %                                 S_2213;S_2223;S_3312;S_3313;S_3323];   
            %
            % S_cart_21: 2D array containing [S_1111;S_2222;S_3333;S_1212;S_1313;S_2323;S_1122;...
            %                                 S_1133;S_1112;S_1113;S_1123;S_2233;S_2212;S_2213;...
            %                                 S_2223;S_3312;S_3313;S_3323;S_1213;S_1223;S_1323];
            %
            % S_cart_21:  2D array containing the repeated elements of S_cart_15 in the C's format
            Nvox=size(S_cart_15,2);
            S_cart_21=zeros(21,Nvox);
            S_cart_21(1,:)=S_cart_15(1,:);
            S_cart_21(2,:)=S_cart_15(2,:);
            S_cart_21(3,:)=S_cart_15(3,:);
            S_cart_21(4,:)=S_cart_15(4,:);%4-
            S_cart_21(5,:)=S_cart_15(5,:);%5-
            S_cart_21(6,:)=S_cart_15(9,:);%9-
            S_cart_21(7,:)=S_cart_15(4,:);%4-
            S_cart_21(8,:)=S_cart_15(5,:);%5-
            S_cart_21(9,:)=S_cart_15(6,:);
            S_cart_21(10,:)=S_cart_15(7,:);
            S_cart_21(11,:)=S_cart_15(8,:);%8-
            S_cart_21(12,:)=S_cart_15(9,:);%9-
            S_cart_21(13,:)=S_cart_15(10,:);
            S_cart_21(14,:)=S_cart_15(11,:);%11-
            S_cart_21(15,:)=S_cart_15(12,:);
            S_cart_21(16,:)=S_cart_15(13,:);%13-
            S_cart_21(17,:)=S_cart_15(14,:);
            S_cart_21(18,:)=S_cart_15(15,:);
            S_cart_21(19,:)=S_cart_15(8,:);%8-
            S_cart_21(20,:)=S_cart_15(11,:);%11-
            S_cart_21(21,:)=S_cart_15(13,:);%13-
        end
        % =================================================================  
        function A_rank4 = map_Arank2_to_Arank4_cartesian(A_rank2)
            % A_rank4 = map_Arank2_to_Arank4_cartesian(A_rank2)
            %
            % A_rank2: 2D array containing [A_11;A_22;A_33;A_12;A_13;A_23]
            %
            % A_rank4: 2D array containing [A_1111;A_2222;A_3333;A_1212;A_1313;A_2323;A_1122;...
            %                               A_1133;A_1112;A_1113;A_1123;A_2233;A_2212;A_2213;...
            %                               A_2223;A_3312;A_3313;A_3323;A_1213;A_1223;A_1323];
            trA = A_rank2(1,:) + A_rank2(2,:) + A_rank2(3,:);
            idx=[ 1,1,1,1; 2,2,2,2; 3,3,3,3; 1,2,1,2; 1,3,1,3; 2,3,2,3; 1,1,2,2;...
                  1,1,3,3; 1,1,1,2; 1,1,1,3; 1,1,2,3; 2,2,3,3; 2,2,1,2; 2,2,1,3;...
                  2,2,2,3; 3,3,1,2; 3,3,1,3; 3,3,2,3; 1,2,1,3; 1,2,2,3; 1,3,2,3];
            delta=eye(3);
            A_rank2_3D(1,1,:)=A_rank2(1,:);
            A_rank2_3D(1,2,:)=A_rank2(4,:);
            A_rank2_3D(1,3,:)=A_rank2(5,:);
            A_rank2_3D(2,1,:)=A_rank2(4,:);
            A_rank2_3D(2,2,:)=A_rank2(2,:);
            A_rank2_3D(2,3,:)=A_rank2(6,:);
            A_rank2_3D(3,1,:)=A_rank2(5,:);
            A_rank2_3D(3,2,:)=A_rank2(6,:);
            A_rank2_3D(3,3,:)=A_rank2(3,:);
            for id=1:size(idx,1)
                ii=idx(id,1); jj=idx(id,2); kk=idx(id,3); ll=idx(id,4);
                term1(id,:) = 2*delta(ii,jj)*delta(kk,ll) - delta(ii,ll)*delta(jj,kk) - delta(ii,kk)*delta(jj,ll);
                term2(id,:) = delta(ii,jj)*A_rank2_3D(kk,ll,:) + delta(kk,ll)*A_rank2_3D(ii,jj,:);
                term3(id,:) = delta(jj,kk)*A_rank2_3D(ii,ll,:) + delta(ii,ll)*A_rank2_3D(jj,kk,:) + delta(ii,kk)*A_rank2_3D(jj,ll,:) + delta(jj,ll)*A_rank2_3D(ii,kk,:);
            end
            A_rank4 = 1/6*( trA.*term1 - 2*term2 + term3 );
        end
        % =================================================================  
        function A_rank2 = map_Arank4_to_Arank2_cartesian(A_rank4) % updated Sept2024 for Racah normalization
            % A_rank2 = map_Arank4_to_Arank2_cartesian(A_rank4)
            %
            % A_rank4: 2D array containing [A_1111;A_2222;A_3333;A_1212;A_1313;A_2323;A_1122;...
            %                               A_1133;A_1112;A_1113;A_1123;A_2233;A_2212;A_2213;...
            %                               A_2223;A_3312;A_3313;A_3323;A_1213;A_1223;A_1323];
            %
            % A_rank2: 2D array containing [A_11;A_22;A_33;A_12;A_13;A_23]
            A_rank2=A_rank4(1:6,:)*0;
            A_rank2(1,:) = 2*( A_rank4(12,:) - A_rank4(6,:) );
            A_rank2(2,:) = 2*( A_rank4(8,:) - A_rank4(5,:) );
            A_rank2(3,:) = 2*( A_rank4(7,:) - A_rank4(4,:) );
            A_rank2(4,:) = 2*( A_rank4(21,:) - A_rank4(16,:) );
            A_rank2(5,:) = 2*( A_rank4(20,:) - A_rank4(14,:) );
            A_rank2(6,:) = 2*( A_rank4(19,:) - A_rank4(11,:) );
        end
        % =================================================================
        function [DlmDlm_real] = DlmDlm_complex2real_STF(DlmDlm_complex)
            % [DlmDlm_real] = DlmDlm_complex2real_STF(DlmDlm_complex)
            %
            % DlmDlm_complex and DlmDlm_real: 2D arrays [21 x M] containing complex and real STF of
            % [D00D00,D00D2m2,D00D2m1,D00D20,D00D21,D00D22,D2m2D2m2,D2m2D2m1,D2m2D20,D2m2D21,D2m2D22,D2m1D2m1,D2m1D20,D2m1D21,D2m1D22,D20D20,D20D21,D20D22,D21D21,D21D22,D22D22];
            %
            % S_cart_21:  2D array containing the repeated elements of S_cart_15 in the C's format
            if isvector(DlmDlm_complex)
                vector_flag = 1;
                DlmDlm_complex = repmat(DlmDlm_complex(:),1,5);
            else
                vector_flag = 0;
            end
            DlmDlm_real = zeros(size(DlmDlm_complex));
            % [D00D00,D00D2m2,D00D2m1,D00D20,D00D21,D00D22,D2m2D2m2,D2m2D2m1,D2m2D20,D2m2D21,D2m2D22,D2m1D2m1,D2m1D20,D2m1D21,D2m1D22,D20D20,D20D21,D20D22,D21D21,D21D22,D22D22];

            % [D00D00,~,~,D00D20,~,~,~,~,~,~,~,~,~,~,~,D20D20,~,~,~,~,~]
            DlmDlm_real([1 4 16],:) = DlmDlm_complex([1 4 16],:);
            % [~,D00D2m2,D00D2m1,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~];        
            DlmDlm_real([2 3],:) = sqrt(2) * imag(DlmDlm_complex([2 3],:));
            % [~,~,~,~,D00D21,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~];        
            DlmDlm_real(5,:) = - sqrt(2) * real(DlmDlm_complex(5,:));
            % [~,~,~,~,~,D00D22,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~];        
            DlmDlm_real(6,:) = sqrt(2) * real(DlmDlm_complex(6,:));
            % [~,~,~,~,~,~,D2m2D2m2,~,~,~,~,~,~,~,~,~,~,~,~,~,~];
            DlmDlm_real(7,:) =  - real(DlmDlm_complex(7,:))  + real(DlmDlm_complex(11,:));
            % [~,~,~,~,~,~,~,D2m2D2m1,~,~,~,~,~,~,~,~,~,~,~,~,~];
            DlmDlm_real(8,:) =  - real(DlmDlm_complex(8,:))  - real(DlmDlm_complex(10,:));
            % [~,~,~,~,~,~,~,~,~,~,~,D2m1D2m1,~,~,~,~,~,~,~,~,~];
            DlmDlm_real(12,:) = - real(DlmDlm_complex(12,:)) - real(DlmDlm_complex(14,:));
            % [~,~,~,~,~,~,~,~,D2m2D20,D2m1D20,~,~,~,~,~,~,~,~,~,~,~];
            DlmDlm_real([9 13],:) = sqrt(2) * imag(DlmDlm_complex([9 13],:));
            % [~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,D20D21,~,~,~,~];
            DlmDlm_real(17,:) = - sqrt(2) * real(DlmDlm_complex(17,:));
            % [~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,D20D22,~,~,~];
            DlmDlm_real(18,:) = sqrt(2) * real(DlmDlm_complex(18,:));
            % [~,~,~,~,~,~,~,~,~,D2m2D21,~,~,~,D2m1D21,~,~,~,~,~,~,~];
            DlmDlm_real([10 14],:) = - imag(DlmDlm_complex([10 14],:)) + imag(DlmDlm_complex([8 12],:));
            % [~,~,~,~,~,~,~,~,~,~,D2m2D22,~,~,~,D2m1D22,~,~,~,~,~,~];
            DlmDlm_real([11 15],:) =   imag(DlmDlm_complex([11 15],:)) + imag(DlmDlm_complex([7 8],:));
            % [~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,D21D21,~,~];
            DlmDlm_real(19,:) = real(DlmDlm_complex(19,:)) - real(DlmDlm_complex(14,:));
            % [~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,D21D22,~];
            DlmDlm_real(20,:) = -real(DlmDlm_complex(20,:)) - real(DlmDlm_complex(10,:));
            % [~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,D22D22];
            DlmDlm_real(21,:) = real(DlmDlm_complex(21,:)) + real(DlmDlm_complex(11,:));
            if vector_flag
                DlmDlm_real = DlmDlm_real(:,1);
            end
        end
        % =================================================================
        function [Slm,Alm] = TQ2SA(Tlm,Qlm) % updated Sept2024 for Racah normalization
            % [Slm,Alm] = TQ2SA(Tlm,Qlm)
            %
            % Tlm: 2D array [15 x M] containing [T_00;T_2-2;T_2-1;T_20;T_21;T_22;T_4-4;T_4-3;T_4-2;T_4-1;T_40;T_41;T_42;T_43;T_44]; 
            %
            % Qlm: 2D array [6 x M] containing [Q_00;Q_2-2;Q_2-1;Q_20;Q_21;Q_22]
            %
            % Slm: 2D array [15 x M] containing [S_00;S_2-2;S_2-1;S_20;S_21;S_22;S_4-4;S_4-3;S_4-2;S_4-1;S_40;S_41;S_42;S_43;S_44]; 
            %
            % Alm: 2D array [6 x M] containing [A_00;A_2-2;A_2-1;A_20;A_21;A_22]
            if isvector(Tlm)
                only_one_tensor = 1;
                Tlm = repmat(Tlm(:),1,5);
                Qlm = repmat(Qlm(:),1,5);
            else
                only_one_tensor = 0;
            end
            S00 = Qlm(1,:) + Tlm(1,:) ;
            A00 = 2 * Qlm(1,:) - 5/2 * Tlm(1,:) ;
            S2m = Qlm(2:6,:) + Tlm(2:6,:) ;
            A2m = - Qlm(2:6,:) + 7/2 * Tlm(2:6,:) ;
            S4m = Tlm(7:15,:);
            Slm = [S00 ; S2m ; S4m];
            Alm = [A00 ; A2m ];
            if only_one_tensor
                Slm = Slm(:,1);
                Alm = Alm(:,1);
            end
        end
        % =================================================================
        function [Tlm,Qlm] = SA2TQ(Slm,Alm) % updated Sept2024 for Racah normalization
            % [Tlm,Qlm] = SA2TQ(Slm,Alm)
            %
            % Slm: 2D array [15 x M] containing [S_00;S_2-2;S_2-1;S_20;S_21;S_22;S_4-4;S_4-3;S_4-2;S_4-1;S_40;S_41;S_42;S_43;S_44]; 
            %
            % Alm: 2D array [6 x M] containing [A_00;A_2-2;A_2-1;A_20;A_21;A_22]
            %
            % Tlm: 2D array [15 x M] containing [T_00;T_2-2;T_2-1;T_20;T_21;T_22;T_4-4;T_4-3;T_4-2;T_4-1;T_40;T_41;T_42;T_43;T_44]; 
            %
            % Qlm: 2D array [6 x M] containing [Q_00;Q_2-2;Q_2-1;Q_20;Q_21;Q_22]
            if isvector(Slm)
                only_one_tensor = 1;
                Slm = repmat(Slm(:),1,5);
                Alm = repmat(Alm(:),1,5);
            else
                only_one_tensor = 0;
            end
            Q00 = 1/9 * (2 * Alm(1,:)    + 5 * Slm(1,:));
            Q2m = 1/9 * (-2 * Alm(2:6,:) + 7 * Slm(2:6,:));
            T00 = 1/9 * (-2 * Alm(1,:)   + 4 * Slm(1,:));
            T2m = 1/9 * ( 2 * Alm(2:6,:) + 2 * Slm(2:6,:));
            T4m = Slm(7:15,:);
            Tlm = [T00 ; T2m ; T4m];
            Qlm = [Q00 ; Q2m ];
            if only_one_tensor
                Tlm = Tlm(:,1);
                Qlm = Qlm(:,1);
            end
        end
        % =================================================================
        function [DlmDlm, D2mD2m_5x5] = TQ2DD_complexSTF(Tlm,Qlm) % updated Sept2024 for Racah normalization
            % [DlmDlm, D2mD2m_5x5] = TQ2DD_complexSTF(Tlm,Qlm)
            %
            % Tlm: 2D array [15 x M] containing [T_00;T_2-2;T_2-1;T_20;T_21;T_22;T_4-4;T_4-3;T_4-2;T_4-1;T_40;T_41;T_42;T_43;T_44]; 
            %
            % Qlm: 2D array [6 x M] containing [Q_00;Q_2-2;Q_2-1;Q_20;Q_21;Q_22];
            %
            % DlmDlm: 2D array [21 x M] containing [D00D00 ; D00D2m2 ; D00D2m1 ; D00D20 ; D00D21 ; D00D22 ; D2m2D2m2 ; D2m2D2m1 ; D2m2D20 ; D2m2D21 ; D2m2D22 ; D2m1D2m1 ; D2m1D20 ; D2m1D21 ; D2m1D22 ; D20D20 ; D20D21 ; D20D22 ; D21D21 ; D21D22 ; D22D22];
            if isvector(Tlm)
                only_one_tensor = 1;
                Tlm = repmat(Tlm(:),1,8);
                Qlm = repmat(Qlm(:),1,8);
            else
                only_one_tensor = 0;
            end
            D00D00 =  Qlm(1,:); % checked
            D00D2m2 = 1/2 * Qlm(2,:); % checked
            D00D2m1 = 1/2 * Qlm(3,:); % checked
            D00D20  = 1/2 * Qlm(4,:); % checked
            D00D21  = 1/2 * Qlm(5,:); % checked
            D00D22  = 1/2 * Qlm(6,:); % checked
            



T00  = Tlm(1,:);
T2m2 = Tlm(2,:);
T2m1 = Tlm(3,:);
T20  = Tlm(4,:);
T21  = Tlm(5,:);
T22  = Tlm(6,:);
T4m4 = Tlm(7,:);
T4m3 = Tlm(8,:);
T4m2 = Tlm(9,:);
T4m1 = Tlm(10,:);
T40  = Tlm(11,:);
T41  = Tlm(12,:);
T42  = Tlm(13,:);
T43  = Tlm(14,:);
T44  = Tlm(15,:);
% D2m2D2m2 = (70^(1/2)*T4m4)/6 ;
% D2m2D2m1 = (35^(1/2)*T4m3)/6 ;
% D2m2D20 = (15^(1/2)*T4m2)/6 - T2m2 ;
% D2m2D21 = (5^(1/2)*T4m1)/6 - (6^(1/2)*T2m1)/2 ;
% D2m2D22 = T00 - T20 + T40/6 ;
% D2m1D2m1 = 10^(1/2)*(T4m2/3 + (15^(1/2)*T2m2)/10) ;
% D2m1D20 = T2m1/2 + (30^(1/2)*T4m1)/6 ;
% D2m1D21 = (2*T40)/3 - T20/2 - T00 ;
% D2m1D22 = (5^(1/2)*T41)/6 - (6^(1/2)*T21)/2 ;
% D20D20 = T00 + T20 + T40 ;
% D20D21 = T21/2 + (30^(1/2)*T41)/6 ;
% D20D22 = (15^(1/2)*T42)/6 - T22 ;
% D21D21 = 10^(1/2)*(T42/3 + (15^(1/2)*T22)/10) ;
% D21D22 = (35^(1/2)*T43)/6 ;
% D22D22 = (70^(1/2)*T44)/6 ;

D2m2D2m2 = (70^(1/2)*T4m4)/6 ;
D2m2D2m1 = (35^(1/2)*T4m3)/6 ;
D2m2D20 = (15^(1/2)*T4m2)/6 - T2m2 ;
D2m2D21 = (5^(1/2)*T4m1)/6 - (6^(1/2)*T2m1)/2 ;
D2m2D22 = T00 - T20 + T40/6 ;
D2m1D2m1 = 10^(1/2)*(T4m2/3 + (15^(1/2)*T2m2)/10) ;
D2m1D20 = T2m1/2 + (30^(1/2)*T4m1)/6 ;
D2m1D21 = (2*T40)/3 - T20/2 - T00 ;
D2m1D22 = (5^(1/2)*T41)/6 - (6^(1/2)*T21)/2 ;
D20D20 = T00 + T20 + T40 ;
D20D21 = T21/2 + (30^(1/2)*T41)/6 ;
D20D22 = (15^(1/2)*T42)/6 - T22 ;
D21D21 = 10^(1/2)*(T42/3 + (15^(1/2)*T22)/10) ;
D21D22 = (35^(1/2)*T43)/6 ;
D22D22 = (70^(1/2)*T44)/6 ;

            DlmDlm = [D00D00 ; D00D2m2 ; D00D2m1 ; D00D20 ; D00D21 ; D00D22 ; D2m2D2m2 ; D2m2D2m1 ; D2m2D20 ; D2m2D21 ; D2m2D22 ; D2m1D2m1 ; D2m1D20 ; D2m1D21 ; D2m1D22 ; D20D20 ; D20D21 ; D20D22 ; D21D21 ; D21D22 ; D22D22];
            D2mD2m_5x5 = [ D2m2D2m2 ; D2m2D2m1 ; D2m2D20 ; D2m2D21 ; D2m2D22 ;...
                           D2m2D2m1 ; D2m1D2m1 ; D2m1D20 ; D2m1D21 ; D2m1D22 ;...
                           D2m2D20  ; D2m1D20  ; D20D20  ; D20D21  ; D20D22 ;...
                           D2m2D21  ; D2m1D21  ; D20D21  ; D21D21  ; D21D22 ;...
                           D2m2D22  ; D2m1D22  ; D20D22  ; D21D22  ; D22D22 ];
            D2mD2m_5x5 = reshape(D2mD2m_5x5,5,5,length(D00D00));
            if only_one_tensor
                DlmDlm = DlmDlm(:,1);
                D2mD2m_5x5 = D2mD2m_5x5(:,:,1);
            end
        end
        % =================================================================
        function [DlmDlm, D2mD2m_5x5] = NumericalDlmDlm(Tlm,Qlm,CSphase,ComplexSTF) % updated Sept2024 for Racah normalization
            % [DlmDlm, D2mD2m_5x5] = NumericalDlmDlm(Tlm,Qlm,CSphase,ComplexSTF)
            %
            % Tlm: 2D array [15 x M] containing [T_00;T_2-2;T_2-1;T_20;T_21;T_22;T_4-4;T_4-3;T_4-2;T_4-1;T_40;T_41;T_42;T_43;T_44]; 
            %
            % Qlm: 2D array [6 x M] containing [Q_00;Q_2-2;Q_2-1;Q_20;Q_21;Q_22];  
            %
            % DlmDlm = [D00D00 ; D00D2m2 ; D00D2m1 ; D00D20 ; D00D21 ; D00D22 ; D2m2D2m2 ; D2m2D2m1 ; D2m2D20 ; D2m2D21 ; D2m2D22 ; D2m1D2m1 ; D2m1D20 ; D2m1D21 ; D2m1D22 ; D20D20 ; D20D21 ; D20D22 ; D21D21 ; D21D22 ; D22D22];
            if ~exist('CSphase','var') || isempty(CSphase) || CSphase
                CSphase=1; % 1 means we use it (default)
            else
                CSphase=0; % 0 means we DO NOT use it
            end
            if ~exist('ComplexSTF','var') || isempty(ComplexSTF) || ~ComplexSTF
                ComplexSTF=0; % 0 means we use real STF basis (default)
            else
                ComplexSTF=1; % 1 means we use complex STF basis
            end
            if isvector(Tlm)
                only_one_tensor = 1;
                Tlm = repmat(Tlm(:),1,8);
                Qlm = repmat(Qlm(:),1,8);
            else
                only_one_tensor = 0;
            end
            % C0=sqrt(1/(4*pi)); C2=sqrt(5/(4*pi)); C4=sqrt(9/(4*pi));
            % Fixing Racah's normalization
            flattened = 0;
            if ComplexSTF
                Y2_complex = RICEtools.get_STF_basis(2,CSphase,1,flattened);
                Y4_complex = RICEtools.get_STF_basis(4,CSphase,1,flattened);
                for ii=1:5
                    for jj=1:5
                        for kk=1:5
                            % Y2Y2Y2_complex(ii,jj,kk) = trace(conj(Y2_complex(:,:,ii+1))*conj(Y2_complex(:,:,jj+1))*Y2_complex(:,:,kk+1));
                            Y2Y2Y2_complex(ii,jj,kk) = trace((Y2_complex(:,:,ii+1))*(Y2_complex(:,:,jj+1))*conj(Y2_complex(:,:,kk+1)));
                        end
                    end
                end
                for ii=1:5
                    for jj=1:5
                        for kk=1:9
                            % Y2Y2Y4_complex(ii,jj,kk)    = tensorprod(tensorprod(conj(Y2_complex(:,:,ii+1)),conj(Y2_complex(:,:,jj+1))),Y4_complex(:,:,:,:,kk+6),1:4);
                            Y2Y2Y4_complex(ii,jj,kk)    = tensorprod(tensorprod((Y2_complex(:,:,ii+1)),(Y2_complex(:,:,jj+1))),conj(Y4_complex(:,:,:,:,kk+6)),1:4);
                        end
                    end
                end
                Y2Y2Y2_x = Y2Y2Y2_complex;
                Y2Y2Y4_x = Y2Y2Y4_complex;
            else
                Y2_real = RICEtools.get_STF_basis(2,CSphase,0,flattened);
                Y4_real = RICEtools.get_STF_basis(4,CSphase,0,flattened);
                for ii=1:5
                    for jj=1:5
                        for kk=1:5
                            Y2Y2Y2_real(ii,jj,kk)    = trace(Y2_real(:,:,ii+1)*Y2_real(:,:,jj+1)*Y2_real(:,:,kk+1));
                        end
                    end
                end
                for ii=1:5
                    for jj=1:5
                        for kk=1:9
                            Y2Y2Y4_real(ii,jj,kk)    = tensorprod(tensorprod(Y2_real(:,:,ii+1),Y2_real(:,:,jj+1)),Y4_real(:,:,:,:,kk+6),1:4);
                        end
                    end
                end
                Y2Y2Y2_x = Y2Y2Y2_real;
                Y2Y2Y4_x = Y2Y2Y4_real;
            end
            % Creating full system
            ids = [1:5 7:10 13:15 19:20 25];
            count = [1 2 2 2 2 1 2 2 2 1 2 2 1 2 1];
            if ComplexSTF
                YY_15x5(1,:)    = [ 0 , 0 , 0 , 0 , 2 , 0 , 0 , -2 , 0 , 1  , 0  , 0 , 0  , 0 , 0 ]/5;
            else
                YY_15x5(1,:)    = [ 1 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1  , 0  , 0 , 1  , 0 , 1 ]/5;
            end
            for ii=1:5
                aux = Y2Y2Y2_x(:,:,ii);
                YY_15x5(ii+1,:)  = 8/21 * aux(ids).*count;
            end
            for ii=1:9
                aux = Y2Y2Y4_x(:,:,ii);
                YY_15x5(ii+6,:)  = 8/35 * aux(ids).*count;
            end
%             D2mD2m_5x5_unique_pinv = YY_15x5\Tlm;
            D2mD2m_5x5_unique_pinv = inv(YY_15x5)*Tlm;
            D00D00 = Qlm(1,:);
            D00D2m2 = 1/2 * Qlm(2,:);
            D00D2m1 = 1/2 * Qlm(3,:);
            D00D20  = 1/2 * Qlm(4,:);
            D00D21  = 1/2 * Qlm(5,:);
            D00D22  = 1/2 * Qlm(6,:);
            D2m2D2m2 = D2mD2m_5x5_unique_pinv(1,:) ;
            D2m2D2m1 = D2mD2m_5x5_unique_pinv(2,:) ;
            D2m2D20  = D2mD2m_5x5_unique_pinv(3,:) ;
            D2m2D21  = D2mD2m_5x5_unique_pinv(4,:) ;
            D2m2D22  = D2mD2m_5x5_unique_pinv(5,:) ;
            D2m1D2m1 = D2mD2m_5x5_unique_pinv(6,:) ;
            D2m1D20  = D2mD2m_5x5_unique_pinv(7,:) ;
            D2m1D21  = D2mD2m_5x5_unique_pinv(8,:) ;
            D2m1D22  = D2mD2m_5x5_unique_pinv(9,:) ;
            D20D20   = D2mD2m_5x5_unique_pinv(10,:) ;
            D20D21   = D2mD2m_5x5_unique_pinv(11,:) ;
            D20D22   = D2mD2m_5x5_unique_pinv(12,:) ;
            D21D21   = D2mD2m_5x5_unique_pinv(13,:) ;
            D21D22   = D2mD2m_5x5_unique_pinv(14,:) ;
            D22D22   = D2mD2m_5x5_unique_pinv(15,:) ;
            DlmDlm = [D00D00 ; D00D2m2 ; D00D2m1 ; D00D20 ; D00D21 ; D00D22 ; D2m2D2m2 ; D2m2D2m1 ; D2m2D20 ; D2m2D21 ; D2m2D22 ; D2m1D2m1 ; D2m1D20 ; D2m1D21 ; D2m1D22 ; D20D20 ; D20D21 ; D20D22 ; D21D21 ; D21D22 ; D22D22];
            D2mD2m_5x5 = [ D2m2D2m2 ; D2m2D2m1 ; D2m2D20 ; D2m2D21 ; D2m2D22 ;...
                           D2m2D2m1 ; D2m1D2m1 ; D2m1D20 ; D2m1D21 ; D2m1D22 ;...
                           D2m2D20  ; D2m1D20  ; D20D20  ; D20D21  ; D20D22 ;...
                           D2m2D21  ; D2m1D21  ; D20D21  ; D21D21  ; D21D22 ;...
                           D2m2D22  ; D2m1D22  ; D20D22  ; D21D22  ; D22D22 ];
            D2mD2m_5x5 = reshape(D2mD2m_5x5,5,5,length(D2m2D2m2));
            if only_one_tensor
                DlmDlm = DlmDlm(:,1);
                D2mD2m_5x5 = D2mD2m_5x5(:,:,1);
            end
        end
        % =================================================================
        function [Tlm,Qlm] = DlmDlm_2_TQ(DlmDlm,CSphase,ComplexSTF) % updated Sept2024 for Racah normalization
            % [Tlm,Qlm] = DlmDlm_2_TQ(DlmDlm,CSphase,ComplexSTF)
            %
            % DlmDlm = [D00D00 ; D00D2m2 ; D00D2m1 ; D00D20 ; D00D21 ; D00D22 ; D2m2D2m2 ; D2m2D2m1 ; D2m2D20 ; D2m2D21 ; D2m2D22 ; D2m1D2m1 ; D2m1D20 ; D2m1D21 ; D2m1D22 ; D20D20 ; D20D21 ; D20D22 ; D21D21 ; D21D22 ; D22D22];
            %
            % Tlm: 2D array [15 x M] containing [T_00;T_2-2;T_2-1;T_20;T_21;T_22;T_4-4;T_4-3;T_4-2;T_4-1;T_40;T_41;T_42;T_43;T_44]; 
            %
            % Qlm: 2D array [6 x M] containing [Q_00;Q_2-2;Q_2-1;Q_20;Q_21;Q_22];  
            if ~exist('CSphase','var') || isempty(CSphase) || CSphase
                CSphase=1; % 1 means we use it (default)
            else
                CSphase=0; % 0 means we DO NOT use it
            end
            if ~exist('ComplexSTF','var') || isempty(ComplexSTF) || ~ComplexSTF
                ComplexSTF=0; % 0 means we use real STF basis (default)
            else
                ComplexSTF=1; % 1 means we use complex STF basis
            end
            if isvector(DlmDlm)
                only_one_tensor = 1;
                DlmDlm = repmat(DlmDlm(:),1,8);
            else
                only_one_tensor = 0;
            end
            AAAA = -sqrt(2/7); BBBB = sqrt(18/35); % Fixing Racah's normalization
            D00D00   = DlmDlm(1,:) ;
            D00D2m2  = DlmDlm(2,:) ;
            D00D2m1  = DlmDlm(3,:) ;
            D00D20   = DlmDlm(4,:) ; 
            D00D21   = DlmDlm(5,:) ; 
            D00D22   = DlmDlm(6,:) ; 
            D2m2D2m2 = DlmDlm(7,:) ;
            D2m2D2m1 = DlmDlm(8,:) ;
            D2m2D20  = DlmDlm(9,:) ;
            D2m2D21  = DlmDlm(10,:) ;
            D2m2D22  = DlmDlm(11,:) ;
            D2m1D2m1 = DlmDlm(12,:) ;
            D2m1D20  = DlmDlm(13,:) ;
            D2m1D21  = DlmDlm(14,:) ;
            D2m1D22  = DlmDlm(15,:) ;
            D20D20   = DlmDlm(16,:) ;
            D20D21   = DlmDlm(17,:) ;
            D20D22   = DlmDlm(18,:) ;
            D21D21   = DlmDlm(19,:) ;
            D21D22   = DlmDlm(20,:) ;
            D22D22   = DlmDlm(21,:) ;
            Q00  =   D00D00 ;
            Q2m2 = 2 * D00D2m2 ;
            Q2m1 = 2 * D00D2m1 ;
            Q20  = 2 * D00D20  ;
            Q21  = 2 * D00D21  ;
            Q22  = 2 * D00D22  ;
            Qlm = [Q00;Q2m2;Q2m1;Q20;Q21;Q22];  
            if ComplexSTF % Use Clebsch-Gordan coefficients
                T00  = 1/sqrt(5) * (2/sqrt(5) * D2m2D22 - 2/sqrt(5) * D2m1D21 + 1/sqrt(5) * D20D20 ); % checked
                T2m2 = AAAA * (2 * sqrt(2/7) * D2m2D20 - sqrt(3/7) * D2m1D2m1); % checked
                T2m1 = AAAA * (2 * sqrt(3/7) * D2m2D21 - 2 * sqrt(1/14) * D2m1D20); % checked
                T20  = AAAA * (2 * sqrt(2/7) * D2m2D22 + 2 * sqrt(1/14) * D2m1D21 - sqrt(2/7) * D20D20); % checked
                T21  = AAAA * (2 * sqrt(3/7) * D2m1D22 - 2 * sqrt(1/14) * D20D21); % checked
                T22  = AAAA * (2 * sqrt(2/7) * D20D22 - sqrt(3/7) * D21D21); % checked
                T4m4 = BBBB * D2m2D2m2 ; % checked
                T4m3 = BBBB *sqrt(2)* D2m2D2m1; % checked
                T4m2 = BBBB * (2 * sqrt(3/14) * D2m2D20 + sqrt(4/7) * D2m1D2m1); % checked
                T4m1 = BBBB * (2 * sqrt(1/14) * D2m2D21 + 2 * sqrt(3/7) * D2m1D20); % checked
                T40  = BBBB * (2 * sqrt(1/70) * D2m2D22 + 2 * sqrt(8/35) * D2m1D21 + sqrt(18/35) * D20D20); % checked
                T41  = BBBB * (2 * sqrt(1/14) * D2m1D22 + 2 * sqrt(3/7) * D20D21); % checked
                T42  = BBBB * (2 * sqrt(3/14) * D20D22 + sqrt(4/7) * D21D21); % checked
                T43  = BBBB *sqrt(2)* D21D22; % checked
                T44  = BBBB * D22D22 ; % checked
                Tlm = [T00;T2m2;T2m1;T20;T21;T22;T4m4;T4m3;T4m2;T4m1;T40;T41;T42;T43;T44]; 
            else % compute numerically once for all voxels
                flattened = 0;
                Y2_real = RICEtools.get_STF_basis(2,CSphase,0,flattened);
                Y4_real = RICEtools.get_STF_basis(4,CSphase,0,flattened);
                for ii=1:5
                    for jj=1:5
                        for kk=1:5
                            Y2Y2Y2_real(ii,jj,kk) = trace(Y2_real(:,:,ii+1)*Y2_real(:,:,jj+1)*Y2_real(:,:,kk+1));
                        end
                    end
                end
                for ii=1:5
                    for jj=1:5
                        for kk=1:9
                            Y2Y2Y4_real(ii,jj,kk) = tensorprod(tensorprod(Y2_real(:,:,ii+1),Y2_real(:,:,jj+1)),Y4_real(:,:,:,:,kk+6),1:4);
                        end
                    end
                end
                % Creating full system
                ids = [1:5 7:10 13:15 19:20 25];
                count = [1 2 2 2 2 1 2 2 2 1 2 2 1 2 1];
                YY_15x5(1,:) = [ 1 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1  , 0  , 0 , 1  , 0 , 1 ];
                for ii=1:5
                    aux = Y2Y2Y2_real(:,:,ii);
                    YY_15x5(ii+1,:) = 8/21 * aux(ids).*count;
                end
                for ii=1:9
                    aux = Y2Y2Y4_real(:,:,ii);
                    YY_15x5(ii+6,:) = 8/35 * aux(ids).*count;
                end
                Tlm = YY_15x5*[D2m2D2m2;D2m2D2m1;D2m2D20;D2m2D21;D2m2D22;D2m1D2m1;D2m1D20;D2m1D21;D2m1D22;D20D20;D20D21;D20D22;D21D21;D21D22;D22D22];
            end
            if only_one_tensor
                Tlm = Tlm(:,1);
                Qlm = Qlm(:,1);
            end
        end
        % =================================================================
        function Y_ell = get_STF_basis(Lmax,CSphase,ComplexSTF,flattened) % updated Sept2024 for Racah normalization
            % Works for Lmax = 2, 4
            % if flattened == 1 (default) then output is a [9x6] matrix for Lmax = 2, and an [81x15] matrix for Lmax = 4
            % if flattened == 0 then output is a [3x3x6] array for Lmax = 2, and an [3x3x3x3x15] matrix for Lmax = 4
            if ~exist('CSphase','var') || isempty(CSphase) || CSphase
                CSphase=1; % 1 means we use it (default)
            else
                CSphase=0; % 0 means we DO NOT use it
            end
            if ~exist('ComplexSTF','var') || isempty(ComplexSTF) || ~ComplexSTF
                ComplexSTF=0; % 0 means we use real STF basis (default)
            else
                ComplexSTF=1; % 1 means we use complex STF basis
            end
            if ~exist('flattened','var') || isempty(flattened) || flattened
                flattened=1; % 1 means output is 2-D (default)
            else
                flattened=0; % 0 means output is (L+1)-D
            end
            delta_ij=eye(3); % Fixing Racah's normalization % C0=sqrt(1/(4*pi));
            if Lmax==2
                % Defininig STF basis
                if ComplexSTF
                    Y2_ij_00 = delta_ij;
                    y = RICEtools.getYcomplex(2);
                    Y2_ij_2m2=y{1};
                    Y2_ij_2m1=y{2};
                    Y2_ij_20 =y{3};
                    Y2_ij_21 =y{4};
                    Y2_ij_22 =y{5};
                else
                    y=RICEtools.getY(2);
                    Y2_ij_00 = delta_ij; % Y_ij_00 =y{1,1,1};
                    Y2_ij_2m2=y{1,2,1};
                    Y2_ij_2m1=y{1,2,2};
                    Y2_ij_20 =y{1,2,3};
                    Y2_ij_21 =y{1,2,4};
                    Y2_ij_22 =y{1,2,5};
                end
                if CSphase
                    Y2_ij_2m1=-Y2_ij_2m1;
                    Y2_ij_21=-Y2_ij_21;
                end
                
                if flattened
                    Y_ell = [Y2_ij_00(:), Y2_ij_2m2(:), Y2_ij_2m1(:), Y2_ij_20(:), Y2_ij_21(:), Y2_ij_22(:)];
                else
                    Y_ell = cat(3, Y2_ij_00 , Y2_ij_2m2 , Y2_ij_2m1 , Y2_ij_20 , Y2_ij_21 , Y2_ij_22 );
                end
            elseif Lmax==4
                y = RICEtools.getY(4);
                if ComplexSTF
                    % Using complex SH
                    % Y4_ijkl_00 = y{2,1,1};
                    y = RICEtools.getYcomplex(4);
                    % Y4_ijkl_00 = RICE.symmetrizeTensor(tensorprod(delta_ij,C0*delta_ij)); % tensorprod is only available in newer MATLAB releases
                    % Y4_ijkl_2m2= RICE.symmetrizeTensor(tensorprod(delta_ij,y{1,1}));
                    % Y4_ijkl_2m1= RICE.symmetrizeTensor(tensorprod(delta_ij,y{1,2}));
                    % Y4_ijkl_20 = RICE.symmetrizeTensor(tensorprod(delta_ij,y{1,3}));
                    % Y4_ijkl_21 = RICE.symmetrizeTensor(tensorprod(delta_ij,y{1,4}));
                    % Y4_ijkl_22 = RICE.symmetrizeTensor(tensorprod(delta_ij,y{1,5}));
                    Y4_ijkl_00 = RICEtools.symmetrizeTensor(reshape(kron(delta_ij(:),delta_ij(:)),[3 3 3 3]));
                    Y4_ijkl_2m2= RICEtools.symmetrizeTensor(reshape(kron(delta_ij(:),y{1,1}),[3 3 3 3]));
                    Y4_ijkl_2m1= RICEtools.symmetrizeTensor(reshape(kron(delta_ij(:),y{1,2}),[3 3 3 3]));
                    Y4_ijkl_20 = RICEtools.symmetrizeTensor(reshape(kron(delta_ij(:),y{1,3}),[3 3 3 3]));
                    Y4_ijkl_21 = RICEtools.symmetrizeTensor(reshape(kron(delta_ij(:),y{1,4}),[3 3 3 3]));
                    Y4_ijkl_22 = RICEtools.symmetrizeTensor(reshape(kron(delta_ij(:),y{1,5}),[3 3 3 3]));
                    Y4_ijkl_4m4= y{2,1};
                    Y4_ijkl_4m3= y{2,2};
                    Y4_ijkl_4m2= y{2,3};
                    Y4_ijkl_4m1= y{2,4};
                    Y4_ijkl_40 = y{2,5};
                    Y4_ijkl_41 = y{2,6};
                    Y4_ijkl_42 = y{2,7};
                    Y4_ijkl_43 = y{2,8};
                    Y4_ijkl_44 = y{2,9};
                else
                    % Using real SH
                    Y4_ijkl_00 = y{2,1,1};
                    Y4_ijkl_2m2= y{2,2,1};
                    Y4_ijkl_2m1= y{2,2,2};
                    Y4_ijkl_20 = y{2,2,3};
                    Y4_ijkl_21 = y{2,2,4};
                    Y4_ijkl_22 = y{2,2,5};
                    Y4_ijkl_4m4= y{2,3,1};
                    Y4_ijkl_4m3= y{2,3,2};
                    Y4_ijkl_4m2= y{2,3,3};
                    Y4_ijkl_4m1= y{2,3,4};
                    Y4_ijkl_40 = y{2,3,5};
                    Y4_ijkl_41 = y{2,3,6};
                    Y4_ijkl_42 = y{2,3,7};
                    Y4_ijkl_43 = y{2,3,8};
                    Y4_ijkl_44 = y{2,3,9};
                end
                if CSphase
                    Y4_ijkl_2m1 = -Y4_ijkl_2m1;
                    Y4_ijkl_21  = -Y4_ijkl_21;
                    Y4_ijkl_4m3 = -Y4_ijkl_4m3;
                    Y4_ijkl_4m1 = -Y4_ijkl_4m1;
                    Y4_ijkl_41  = -Y4_ijkl_41;
                    Y4_ijkl_43  = -Y4_ijkl_43;
                end
                if flattened
                    Y_ell = [ Y4_ijkl_00(:), Y4_ijkl_2m2(:), Y4_ijkl_2m1(:), Y4_ijkl_20(:), Y4_ijkl_21(:), Y4_ijkl_22(:), Y4_ijkl_4m4(:), Y4_ijkl_4m3(:), Y4_ijkl_4m2(:), Y4_ijkl_4m1(:), Y4_ijkl_40(:), Y4_ijkl_41(:), Y4_ijkl_42(:), Y4_ijkl_43(:), Y4_ijkl_44(:) ];
                else
                    Y_ell = cat(5, Y4_ijkl_00 , Y4_ijkl_2m2 , Y4_ijkl_2m1 , Y4_ijkl_20 , Y4_ijkl_21 , Y4_ijkl_22 , Y4_ijkl_4m4 , Y4_ijkl_4m3 , Y4_ijkl_4m2 , Y4_ijkl_4m1 , Y4_ijkl_40 , Y4_ijkl_41 , Y4_ijkl_42 , Y4_ijkl_43 , Y4_ijkl_44 );
                end
            else
                error('Only Lmax = 2 or 4 are supported')
            end
        end
        % =================================================================
        function S = symmetrizeTensorPartial(T,ids) % updated Sept2024 for Racah normalization
            % S = symmetrizeTensorPartial(T,ids)
            %
            % clc,clear, T=randn(3,3,3,3,3,3); ids = [1 4 5];
            %
            % By: Santiago Coelho (03/04/2023)
            if ~exist('ids', 'var') || isempty(ids)
                sz = size(T);
                ids = 1:length(sz);
            end
            ids = sort(ids(:)');
            sz = size(T);
            rank = length(sz);
            ids_original_all = 1:rank;
            ids_tail = (length(ids)+1):rank;
            ids_complement = ids_original_all(~ismember(ids_original_all,ids));
            beggining_permutation = [ids ids_complement];
            reverse_beggining_permutation(beggining_permutation) = 1:rank;
            Tbeggining = permute(T,beggining_permutation);
            Nperm=factorial(length(ids));
            inds=perms(1:length(ids));
            S = zeros(sz);
            p = gcp('nocreate');
            if isempty(p)
                for ii=1:Nperm
                    S = S + permute(Tbeggining,[inds(ii,:) ids_tail]);
                end
            else
                parfor ii=1:Nperm
                    S = S + permute(Tbeggining,[inds(ii,:) ids_tail]);
                end
            end
            S = permute(S,reverse_beggining_permutation)/Nperm;
        end
        % =================================================================
        function S = symmetrizeTensor(T) % updated Sept2024 for Racah normalization
            % S = symmetrizeTensor(T)
            %
            % By: Santiago Coelho (01/04/2023)
            sz = size(T);
            rank = length(sz);
            Nperm=factorial(rank);
            inds=perms(1:rank);
            S = zeros(sz);
            p = gcp('nocreate');
            if isempty(p)
                for ii=1:Nperm
                    S = S + permute(T,inds(ii,:));
                end
            else
                parfor ii=1:Nperm
                    S = S + permute(T,inds(ii,:));
                end
            end
            S = S/Nperm;
        end
        % =================================================================
        function [E_a,lambda_a] = eigTensor_rank4_6x6(s4)
            % [E_a,lambda_a] = eigTensor_rank4_6x6(s4)
            %
            % By: Santiago Coelho (23/02/2023)
            [V,D] = eig(s4);
            lambda_a = diag(D)';
            E_a = zeros(3,3,6);
            for ii=1:6
                vi = V(:,ii)/norm(V(:,ii));
                % Build corresponding eigentensor
                Ei=[vi(1) 1/sqrt(2)*vi(4)  1/sqrt(2)*vi(5) ;  1/sqrt(2)*vi(4) vi(2) 1/sqrt(2)*vi(6) ;  1/sqrt(2)*vi(5) 1/sqrt(2)*vi(6) vi(3)];
                % Enforce non-negative determinant
                dd = det(Ei);
                if dd~=0
                    Ei = Ei*sign(dd);
                end
                E_a(:,:,ii) = Ei;
            end
        end
        % =================================================================
        function Rout = GetRotMatBetweenRandRank4Tensors(Rin,S4)
            % Rout = GetRotMatBetweenRandRank4Tensors(Rin,S4)
            %
            % Extract coordinate basis from rank-4 tensor and compute rotation matrix
            % 'Rout' to reference rotation matrix 'Rin'
            %
            % By: Santiago Coelho (03/02/2023)
            s4 = RICEtools.MapRank4_to_6x6(S4,'FullySymmetric');
            [V,D] = eig(s4);
            
            % Ensure det is +1
            V = V*det(V);
            
            % Pick largest eigenvalue of rank-4 tensor (same for any rotation)
            [~,id_max]=max(diag(D));
            
            % Get main eigentensor
            vi=V(:,id_max)/norm(V(:,id_max));
            
            % Build corresponding eigentensor
            E6=[vi(1) 1/sqrt(2)*vi(4)  1/sqrt(2)*vi(5) ;  1/sqrt(2)*vi(4) vi(2) 1/sqrt(2)*vi(6) ;  1/sqrt(2)*vi(5) 1/sqrt(2)*vi(6) vi(3)];
            E6 = E6*sign(det(E6));
            [Vi,~] = eig(E6);
            
            % Ensure det is +1
            Vi = Vi*sign(det(Vi));
            
            % Computing rotation matrix
            Rout = Rin.'*Vi;
            
            % Ensuring the dot product is mostly positive (sign convention used)
            signs = 1-2*(sum(Rout<0)>=2);
            Rout = Rout .* signs;
        end
        % =================================================================
            function RotationalInvariants = ComputeInvariantsFromCumulants(cumulant,type,mask,CSphase,ComplexSTF)
            % RotationalInvariants = ComputeInvariantsFromCumulants(cumulant,type,mask,CSphase,ComplexSTF)
            %
            % cumulant can be a 2D or 4D array with
            % - D(Dlm), A(Alm), Q(Alm) elements in stf basis    , type = 'D', 'A' or 'Q'
            % - S(Slm) or T(Tlm) elements in stf basis          , type = 'S' or 'T'
            % - C(Slm,Alm) or C(Tlm,Qlm) elements in stf basis  , type = 'C'
            %
            % if the array is 4D then mask (which should be 3D) selects which voxels on
            % which this operation is computed
            %
            % Santiago Coelho 01/12/2024
            sz_Slm=size(cumulant);
            if length(sz_Slm)==4
                flag_4D=1;
            elseif length(sz_Slm)==2
                flag_4D=0;
            else
                error('Slm must be a 2D or 4D array')
            end
            if ~exist('CSphase','var') || isempty(CSphase) || CSphase
                CSphase=1; % 1 means we use it (default)
            else
                CSphase=0; % 0 means we DO NOT use it
            end
            if ~exist('ComplexSTF','var') || isempty(ComplexSTF) || ~ComplexSTF
                ComplexSTF=0; % 0 means we use real STF basis (default)
            else
                ComplexSTF=1; % 1 means we use complex STF basis
            end
            
            if flag_4D
                if isempty(mask)
                    mask=true(sz_Slm(1:3));
                end
                Slm_2D = RICEtools.vectorize(cumulant,mask);
                Nlm = sz_Slm(end);
            else
                Slm_2D = cumulant;
                Nlm = sz_Slm(1);
            end
                        
            Nvoxels = size(Slm_2D,2);
            if strcmp(type,'D')||strcmp(type,'A')||strcmp(type,'Q')
                % Compute invariants of rank-2 symmetric tensor: from 6 dof, 3 invariants = 3 intrinsic
                S00 = Slm_2D(1,:);
                S2m = Slm_2D(2:6,:);
                S_0 = S00(1,:);
                S_22 = (S2m(1,:).^2 + S2m(2,:).^2 + S2m(3,:).^2 + S2m(4,:).^2 + S2m(5,:).^2) ;
                S_23 = 1/4 * (- 3*3^(1/2)*S2m(5,:).*S2m(2,:).^2 + 6*3^(1/2)*S2m(4,:).*S2m(2,:).*S2m(1,:) + 3*3^(1/2)*S2m(4,:).^2.*S2m(5,:) - 6*S2m(3,:).*S2m(1,:).^2 + 3*S2m(3,:).*S2m(2,:).^2 - 6*S2m(3,:).*S2m(5,:).^2 + 3*S2m(3,:).*S2m(4,:).^2 + 2*S2m(3,:).^3) ;
                if flag_4D
                    S_0 = RICEtools.vectorize(S_0,mask);
                    S_22 = RICEtools.vectorize(S_22,mask);
                    S_23 = RICEtools.vectorize(S_23,mask);
                end   
            elseif strcmp(type,'S')
                % Compute invariants of rank-4 symmetric tensor: from 15 dof, 12 invariants = 9 intrinsic + 3 mixed
                S00 = Slm_2D(1,:);
                S2m = Slm_2D(2:6,:);
                S4m = Slm_2D(7:15,:);
                % ell = 0 intrinsic invariants
                S_0 = S00(1,:);
                % ell = 2 intrinsic invariants
                S_22 = (S2m(1,:).^2 + S2m(2,:).^2 + S2m(3,:).^2 + S2m(4,:).^2 + S2m(5,:).^2) ;
                S_23 = 1/4 * (- 3*3^(1/2)*S2m(5,:).*S2m(2,:).^2 + 6*3^(1/2)*S2m(4,:).*S2m(2,:).*S2m(1,:) + 3*3^(1/2)*S2m(4,:).^2.*S2m(5,:) - 6*S2m(3,:).*S2m(1,:).^2 + 3*S2m(3,:).*S2m(2,:).^2 - 6*S2m(3,:).*S2m(5,:).^2 + 3*S2m(3,:).*S2m(4,:).^2 + 2*S2m(3,:).^3) ;
                % ell = 4 intrinsic invariants and mixed invariants
                S_42 = 0 * S_0;
                S_43 = 0 * S_0;
                S_44 = 0 * S_0;
                S_45 = 0 * S_0;
                S_E  = 0 * S_0;
                S_Et = 0 * S_0;
                RS2S4_phi = 0 * S_0;
                RS2S4_theta = 0 * S_0;
                RS2S4_psi = 0 * S_0;
                parfor ii=1:Nvoxels
                    S2 = RICEtools.BuildSTF(S2m(:,ii),2,CSphase,ComplexSTF);
                    S4 = RICEtools.F(S4m(:,ii),4,CSphase,ComplexSTF);
                    s4 = RICEtools.MapRank4_to_6x6(S4,'FullySymmetric');
                    S_22(ii) = trace(S2^2);
                    S_23(ii) = trace(S2^3);
                    S_42(ii) = trace(s4^2);
                    S_43(ii) = trace(s4^3);
                    S_44(ii) = trace(s4^4);
                    S_45(ii) = trace(s4^5);
                    [E_a,lambda_a] = RICEtools.eigTensor_rank4_6x6(s4);
                    E=sum(E_a,3);        
                    w_lamb = permute(repmat(lambda_a(:),[1,3,3]),[2 3 1]);
                    Et=sum(E_a.*w_lamb,3);
                    S_E(ii) = trace(E^3);
                    S_Et(ii) = trace(Et^3);
                    [V2,~] = eig(S2);
                    RS2 = V2*det(V2);
                    RS2S4 = RICEtools.GetRotMatBetweenRandRank4Tensors(RS2,S4);
                    % AXANG = rotm2axang(Rout);
                    eul_S4 = rotm2eul(RS2S4);
                    RS2S4_phi(ii)   = eul_S4(1);
                    RS2S4_theta(ii) = eul_S4(2);
                    RS2S4_psi(ii)   = eul_S4(3);
                end
                if flag_4D
                    S_0  = RICEtools.vectorize(S_0,mask);
                    S_22 = RICEtools.vectorize(S_22,mask);
                    S_23 = RICEtools.vectorize(S_23,mask);
                    S_42 = RICEtools.vectorize(S_42,mask);
                    S_43 = RICEtools.vectorize(S_43,mask);
                    S_44 = RICEtools.vectorize(S_44,mask);
                    S_45 = RICEtools.vectorize(S_45,mask);
                    S_E  = RICEtools.vectorize(S_E,mask);
                    S_Et = RICEtools.vectorize(S_Et,mask);
                    RS2S4_phi = RICEtools.vectorize(RS2S4_phi,mask);
                    RS2S4_theta = RICEtools.vectorize(RS2S4_theta,mask);
                    RS2S4_psi = RICEtools.vectorize(RS2S4_psi,mask);
                end 
            elseif strcmp(type,'C')
                % Compute invariants of covariance tensor: from 21 dof, 18 invariants = 12 intrinsic + 6 mixed
                S00 = Slm_2D(1,:);
                S2m = Slm_2D(2:6,:);
                S4m = Slm_2D(7:15,:);
                A00 = Slm_2D(16,:);
                A2m = Slm_2D(17:21,:);
                % ell = 0 intrinsic invariants
                S_0 = S00(1,:);
                A_0 = A00(1,:);
                % ell = 2 intrinsic invariants
                S_22 = (S2m(1,:).^2 + S2m(2,:).^2 + S2m(3,:).^2 + S2m(4,:).^2 + S2m(5,:).^2) ;
                S_23 = 1/4 * (- 3*3^(1/2)*S2m(5,:).*S2m(2,:).^2 + 6*3^(1/2)*S2m(4,:).*S2m(2,:).*S2m(1,:) + 3*3^(1/2)*S2m(4,:).^2.*S2m(5,:) - 6*S2m(3,:).*S2m(1,:).^2 + 3*S2m(3,:).*S2m(2,:).^2 - 6*S2m(3,:).*S2m(5,:).^2 + 3*S2m(3,:).*S2m(4,:).^2 + 2*S2m(3,:).^3) ;
                A_22 = (A2m(1,:).^2 + A2m(2,:).^2 + A2m(3,:).^2 + A2m(4,:).^2 + A2m(5,:).^2) ;
                A_23 = 1/4 * (- 3*3^(1/2)*A2m(5,:).*A2m(2,:).^2 + 6*3^(1/2)*A2m(4,:).*A2m(2,:).*A2m(1,:) + 3*3^(1/2)*A2m(4,:).^2.*A2m(5,:) - 6*A2m(3,:).*A2m(1,:).^2 + 3*A2m(3,:).*A2m(2,:).^2 - 6*A2m(3,:).*A2m(5,:).^2 + 3*A2m(3,:).*A2m(4,:).^2 + 2*A2m(3,:).^3) ;
                S_42 = 0 * S_0;
                S_43 = 0 * S_0;
                S_44 = 0 * S_0;
                S_45 = 0 * S_0;
                S_E  = 0 * S_0;
                S_Et = 0 * S_0;
                RS2S4_phi = 0 * S_0;
                RS2S4_theta = 0 * S_0;
                RS2S4_psi = 0 * S_0;
                RS2A2_phi = 0 * S_0;
                RS2A2_theta = 0 * S_0;
                RS2A2_psi = 0 * S_0;
                parfor ii=1:Nvoxels
                    S2 = RICEtools.BuildSTF(S2m(:,ii),2,CSphase,ComplexSTF);
                    A2 = RICEtools.BuildSTF(A2m(:,ii),2,CSphase,ComplexSTF);
                    S4 = RICEtools.BuildSTF(S4m(:,ii),4,CSphase,ComplexSTF);
                    s4 = RICEtools.MapRank4_to_6x6(S4,'FullySymmetric');
                    
                    A_22(ii) = trace(A2^2);
                    A_23(ii) = trace(A2^3);
            
                    S_22(ii) = trace(S2^2);
                    S_23(ii) = trace(S2^3);
                    S_42(ii) = trace(s4^2);
                    S_43(ii) = trace(s4^3);
                    S_44(ii) = trace(s4^4);
                    S_45(ii) = trace(s4^5);
            
                    [E_a,lambda_a] = RICEtools.eigTensor_rank4_6x6(s4);
                    E=sum(E_a,3)-eye(3)/sqrt(3); % NOT adding identity
                    % E=sum(E_a,3); % Adding identity  
                    w_lamb = permute(repmat(lambda_a(:),[1,3,3]),[2 3 1]);
                    Et=sum(E_a.*w_lamb,3);
                    S_E(ii) = trace(E^3); % NOT adding identity
                    S_Et(ii) = trace(Et^3);
            
                    [V2,~] = eig(S2);
                    RS2 = V2*det(V2);
            
                    RS2S4 = RICEtools.GetRotMatBetweenRandRank4Tensors(RS2,S4);
            %         AXANG = rotm2axang(RS2S4);
                    eul_S4 = rotm2eul(RS2S4);
                    RS2S4_phi(ii)   = eul_S4(1);
                    RS2S4_theta(ii) = eul_S4(2);
                    RS2S4_psi(ii)   = eul_S4(3);
            
                    % Computing rotation matrix
                    [V2a,~] = eig(A2);
                    RA2 = V2a*det(V2a);
                    RS2A2 = RS2.'*RA2;
                    % Ensuring the dot product is mostly positive (sign convention used)
                    signs = 1-2*(sum(RS2A2<0)>=2);
                    RS2A2 = RS2A2 .* signs;
            
                    eul_A2 = rotm2eul(RS2A2);
                    RS2A2_phi(ii)   = eul_A2(1);
                    RS2A2_theta(ii) = eul_A2(2);
                    RS2A2_psi(ii)   = eul_A2(3);
                end
                if flag_4D
                    S_0  = RICEtools.vectorize(S_0,mask);
                    S_22 = RICEtools.vectorize(S_22,mask);
                    S_23 = RICEtools.vectorize(S_23,mask);
                    S_42 = RICEtools.vectorize(S_42,mask);
                    S_43 = RICEtools.vectorize(S_43,mask);
                    S_44 = RICEtools.vectorize(S_44,mask);
                    S_45 = RICEtools.vectorize(S_45,mask);
                    S_E  = RICEtools.vectorize(S_E,mask);
                    S_Et = RICEtools.vectorize(S_Et,mask);
                    RS2S4_phi = RICEtools.vectorize(RS2S4_phi,mask);
                    RS2S4_theta = RICEtools.vectorize(RS2S4_theta,mask);
                    RS2S4_psi = RICEtools.vectorize(RS2S4_psi,mask);
                    A_0  = RICEtools.vectorize(A_0,mask);
                    A_22 = RICEtools.vectorize(A_22,mask);
                    A_23 = RICEtools.vectorize(A_23,mask);
                    RS2A2_phi = RICEtools.vectorize(RS2A2_phi,mask);
                    RS2A2_theta = RICEtools.vectorize(RS2A2_theta,mask);
                    RS2A2_psi = RICEtools.vectorize(RS2A2_psi,mask);
                end 
            end
            
            if strcmp(type,'D')
                RotationalInvariants.D_0=S_0;
                RotationalInvariants.D_22=S_22;
                RotationalInvariants.D_23=S_23;
            elseif strcmp(type,'A')
                RotationalInvariants.A_0=S_0;
                RotationalInvariants.A_22=S_22;
                RotationalInvariants.A_23=S_23;
            elseif strcmp(type,'S')
                RotationalInvariants.S_0=S_0;
                RotationalInvariants.S_22=S_22;
                RotationalInvariants.S_23=S_23;
                RotationalInvariants.S_42=S_42;
                RotationalInvariants.S_43=S_43;
                RotationalInvariants.S_44=S_44;
                RotationalInvariants.S_45=S_45;
                RotationalInvariants.S_E=S_E;
                RotationalInvariants.S_Et=S_Et;
                RotationalInvariants.RS2S4_phi=RS2S4_phi;
                RotationalInvariants.RS2S4_theta=RS2S4_theta;
                RotationalInvariants.RS2S4_psi=RS2S4_psi;
            elseif strcmp(type,'C')
                RotationalInvariants.S_0=S_0;
                RotationalInvariants.S_22=S_22;
                RotationalInvariants.S_23=S_23;
                RotationalInvariants.S_42=S_42;
                RotationalInvariants.S_43=S_43;
                RotationalInvariants.S_44=S_44;
                RotationalInvariants.S_45=S_45;
                RotationalInvariants.S_E=S_E;
                RotationalInvariants.S_Et=S_Et;
                RotationalInvariants.RS2S4_phi=RS2S4_phi;
                RotationalInvariants.RS2S4_theta=RS2S4_theta;
                RotationalInvariants.RS2S4_psi=RS2S4_psi;
                RotationalInvariants.A_0=A_0;
                RotationalInvariants.A_22=A_22;
                RotationalInvariants.A_23=A_23;    
                RotationalInvariants.RS2A2_phi=RS2A2_phi;
                RotationalInvariants.RS2A2_theta=RS2A2_theta;
                RotationalInvariants.RS2A2_psi=RS2A2_psi;
            end   
        end
        % =================================================================
    end
end

