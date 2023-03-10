classdef RICE
    % =====================================================================
    % RICE: Rotational Invariants of the Cumulant Expansion (RICE) class
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
    %  REFERENCES:
    %  - Coelho, S., Fieremans, E., Novikov, D.S., 2022 (ArXiv)
    %

    methods     ( Static = true )
        % =================================================================
        function [b0, tensor_elems, RICE_maps, DIFF_maps] = fit(DWI, b, dirs, bshape, mask, CSphase, type, nls_flag, parallel_flag)
            % [b0, tensor_elems, RICE_maps, DIFF_maps] = fit(DWI, b, dirs, bshape, mask, CSphase, type, nls_flag, parallel_flag)
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
            %                                      maps has: md ad rd fa (these are axial symmetry approximations) 
            %       type can be 'minimalDKI' and then tensor_elem has:   [D00 D2m S00] (7 elem)
            %                                      RICE has: D0 D2 S0
            %                                      maps has: md fa mw
            %       type can be 'minimalDKI_iso' and then tensor_elem has:   [D00 S00] (2 elem)
            %                                      RICE has: D0 S0
            %                                      maps has: md mw 
            %       type can be 'DKI_no_ell4' and then tensor_elem has:   [D00 D2m S00 S2m] (12 elem)
            %                                      RICE has: D0 D2 S0 S2
            %                                      maps has: md ad rd fa mw rw aw (these are axial symmetry approximations assuming W4=0) 
            %       type can be 'fullDKI' and then tensor_elem has:   [D00 D2m S00 S2m S4m] (21 elem)
            %                                      RICE has: D0 D2 S0 S2 S4
            %                                      maps has: md ad rd fa mw rw aw (these are axial symmetry approximations) 
            %       type can be 'minimalRICE' and then tensor_elem has:  [D00 D2m S00 S2m A00(rank2)] (14 elem)
            %                                      RICE has: D0 D2 S0 S2 A0
            %                                      maps has: md ad rd fa mw ufa  ](these are axial symmetry approximations)
            %       type can be 'fullRICE' and then tensor_elem has:  [D00 D2m S00 S2m S4m A00(rank2) A2m(rank2)] (27 elem)
            %                                      RICE has: D0 D2 S0 S2 S4 A0 A2
            %                                      maps has: md ad rd fa mw aw rw ufa d0d2m SSC (these are axial symmetry approximations) 
            %
            %       maps.ad, maps.rd, maps.aw, and maps.rw are approximations assuming axially symmetric tensors (no need to go to fiber basis)
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
                DWI = RICE.nlmsmooth(DWI,mask, 0*mask, smoothlevel);
            end
            
            DWI = RICE.vectorize(DWI, mask);
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
                fprintf('Using the Condon-Shortley phase on the spherical harmonics definition!\n')
            else
                CSphase=0; % 0 means we DO NOT use it
                fprintf('Not using the Condon-Shortley phase on the spherical harmonics definition!\n')
            end
            
            % Defininig STF basis
            Y2 = RICE.get_STF_basis(2,CSphase);
            Y4 = RICE.get_STF_basis(4,CSphase);
            
            % Computing B-tensors
            [Bset,B3x3xN] = RICE.ConstructAxiallySymmetricB(b,bshape,dirs);
            [Bset_2D] = RICE.Generate_BijBkl_2Dset(Bset);
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
            b0 = RICE.vectorize(b0, mask);
            
            % Recover all tensor elements and rotational invariants
            C0=sqrt(1/(4*pi));
            C2=sqrt(5/(4*pi));
            C4=sqrt(9/(4*pi));       
            tensor_elems=RICE.vectorize(dv(2:end,:), mask);
        
            % All approaches extract MD
            D00=tensor_elems(:,:,:,1);
            D0=C0*abs(D00);
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
                S0=C0*abs(S00);
                RICE_maps.S0=S0;
                DIFF_maps.md=D0;
                W0=3*S0./D0.^2;
                DIFF_maps.mw=W0;
            elseif strcmp(type,'minimalDKI')
                D2m=tensor_elems(:,:,:,2:6);
                D2=C2*sqrt(sum(D2m.^2,4));
                S00=tensor_elems(:,:,:,7);
                S0=C0*abs(S00);
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
                S0=C0*abs(S00);
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
                S0=C0*abs(S00);
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
            elseif strcmp(type,'minimalRICE')
                D2m=tensor_elems(:,:,:,2:6);
                D2=C2*sqrt(sum(D2m.^2,4));
                S00=tensor_elems(:,:,:,7);
                S0=C0*abs(S00);
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
                D0sq = D0.^2 + (2*A0)/9 + (5*S0)/9;
                D2sq = -(10*A0)/9+D2.^2+(20*S0)/9;
                ufa_sq=3*D2sq./(4*D0.^2+2*D2sq);
                ufa_sq(ufa_sq(:)<0)=0;
                DIFF_maps.ufa=sqrt(ufa_sq);
            elseif strcmp(type,'fullRICE')
                D2m=tensor_elems(:,:,:,2:6);
                D2=C2*sqrt(sum(D2m.^2,4));
                S00=tensor_elems(:,:,:,7);
                S0=C0*abs(S00);
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
                % Other maps
                D0sq = D0.^2 + (2*A0)/9 + (5*S0)/9;
                D2sq = -(10*A0)/9+D2.^2+(20*S0)/9;
                ufa_sq=3*D2sq./(4*D0.^2+2*D2sq);
                ufa_sq(ufa_sq(:)<0)=0;
                DIFF_maps.ufa=sqrt(ufa_sq);
                % DIFF_maps.d0d2m=sqrt(sum((D00.*D2m+7/(22*C0)*S2m-1/(11*C0)*A2m).^2,4));
                DIFF_maps.d0d2m_singleBra=sqrt(sum((D00.*D2m+7/(22*C0)*S2m-1/(11*C0)*A2m).^2,4));
                DIFF_maps.d0d2m_doubleBra=sqrt(sum((7/(22*C0)*S2m-1/(11*C0)*A2m).^2,4));
                D00_std=(5*S00+2*A00)/(9*C0);
                D2m_std=(2*S00-A00)*10*C0/(9*C2^2);
                D00_std=sqrt(D0sq)/C0;
                D2m_std=sqrt(D2sq)/C2;
                SSC=DIFF_maps.d0d2m_doubleBra./sqrt(abs(D00_std.*D2m_std));
                DIFF_maps.SSC=SSC;
            end    
        end
        % =================================================================
        function [Slm] = cart2STF(Scart,CSphase)
            % Scart: 2D array containing [S_11;S_22;S_33;S_12;S_13;S_23]
            % or [S_1111;S_2222;S_3333;S_1122;S_1133;S_1112;S_1113;S_1123;S_2233;S_2212;S_2213;S_2223;S_3312;S_3313;S_3323];   
            C0=sqrt(1/(4*pi));
            C2=sqrt(5/(4*pi));
            C4=sqrt(9/(4*pi));
            if size(Scart,1)==6
                Y2 = RICE.get_STF_basis(2,CSphase);
                S_cnt=[1 1 1 2 2 2];
                id_indep_elemens=[1 5 9 4 7 8];
                S00 = 1/3*(1/C0^2)*(Y2(id_indep_elemens,1)'.*S_cnt)*Scart;
                S2m = 2/3*(1/C2^2)*(Y2(id_indep_elemens,2:6)'.*S_cnt)*Scart;
                Slm = [S00; S2m];
            elseif size(Scart,1)==15
                Y4 = RICE.get_STF_basis(4,CSphase);
                S_cnt=[1 1 1 6 6 4 4 12 6 4 12 4 12 4 4];
                id_indep_elemens=[1 41 81 37 73 28 55 64 77 32 59 68 36 63 72];
                S00 = 1/5*(1/C0^2)*(Y4(id_indep_elemens,1)'.*S_cnt)*Scart;
                S2m = 4/7*(1/C2^2)*(Y4(id_indep_elemens,2:6)'.*S_cnt)*Scart;
                S4m = 8/35*(1/C4^2)*(Y4(id_indep_elemens,7:15)'.*S_cnt)*Scart;
                Slm = [S00; S2m; S4m];
            else
                error('Only rank-2 and rank-4 fully symmetric tensors are supported')
            end
        end         
        % =================================================================
        function [Scart] = STF2cart(Slm,CSphase)
            % Scart: 2D array containing [S_11;S_22;S_33;S_12;S_13;S_23]
            % or [S_1111;S_2222;S_3333;S_1122;S_1133;S_1112;S_1113;S_1123;S_2233;S_2212;S_2213;S_2223;S_3312;S_3313;S_3323];   
            if size(Slm,1)==6
                Y2 = RICE.get_STF_basis(2,CSphase);
                Scart = Y2*Slm;
                keep = [1 5 9 2 3 6];
                Scart = Scart(keep,:);
            elseif size(Slm,1)==15
                Y4 = RICE.get_STF_basis(4,CSphase);
                Scart = Y4*Slm;
                keep = [1 41 81 37 73 28 55 64 77 32 59 68 36 63 72];
                Scart = Scart(keep,:);
            else
                error('Only rank-2 and rank-4 fully symmetric tensors are supported')
            end
        end        
        % =================================================================
        function [Y_ell] = get_STF_basis(Lmax,CSphase)
            % Works for Lmax = 2, 4
            % Output is a [9x6] matrix for Lmax = 2, and an [81x15] matrix for Lmax = 4
            if Lmax==2
                % Defininig STF basis
                % Y_{ij}^{00}
                delta_ij=eye(3);
                Y2_ij_00=  delta_ij(:)/sqrt(4*pi);
                % Y_{ij}^{2m}
                Y2_ij_2m2= [0;0.546274215296040;0;0.546274215296040;0;0;0;0;0];
                Y2_ij_2m1= -[0;0;0;0;0;0.546274215296040;0;0.546274215296040;0];
                Y2_ij_20=  [-0.315391565252520;0;0;0;-0.315391565252520;0;0;0;0.630783130505040];
                Y2_ij_21=  -[0;0;0.546274215296040;0;0;0;0.546274215296040;0;0];
                Y2_ij_22=  [0.546274215296040;0;0;0;-0.546274215296040;0;0;0;0];
                if CSphase
                    Y2_ij_2m1=-Y2_ij_2m1;
                    Y2_ij_21=-Y2_ij_21;
                end
                Y_ell=[Y2_ij_00, Y2_ij_2m2, Y2_ij_2m1, Y2_ij_20, Y2_ij_21, Y2_ij_22];
            elseif Lmax==4
                % Y_{ijkl}^{00}
                Y4_ijkl_00 = [0.282094791773878;0;0;0;0.0940315972579594;0;0;0;0.0940315972579594;0;0.0940315972579594;0;0.0940315972579594;0;0;0;0;0;0;0;0.0940315972579594;0;0;0;0.0940315972579594;0;0;0;0.0940315972579594;0;0.0940315972579594;0;0;0;0;0;0.0940315972579594;0;0;0;0.282094791773878;0;0;0;0.0940315972579594;0;0;0;0;0;0.0940315972579594;0;0.0940315972579594;0;0;0;0.0940315972579594;0;0;0;0.0940315972579594;0;0;0;0;0;0;0;0.0940315972579594;0;0.0940315972579594;0;0.0940315972579594;0;0;0;0.0940315972579594;0;0;0;0.282094791773878];
                % symmetrized Y_{ij}^{2m} delta_{kl}
                Y4_ijkl_2m2= [0;0.273137107648020;0;0.273137107648020;0;0;0;0;0;0.273137107648020;0;0;0;0.273137107648020;0;0;0;0.0910457025493399;0;0;0;0;0;0.0910457025493399;0;0.0910457025493399;0;0.273137107648020;0;0;0;0.273137107648020;0;0;0;0.0910457025493399;0;0.273137107648020;0;0.273137107648020;0;0;0;0;0;0;0;0.0910457025493399;0;0;0;0.0910457025493399;0;0;0;0;0;0;0;0.0910457025493399;0;0.0910457025493399;0;0;0;0.0910457025493399;0;0;0;0.0910457025493399;0;0;0;0.0910457025493399;0;0.0910457025493399;0;0;0;0;0];
                Y4_ijkl_2m1= [0;0;0;0;0;-0.0910457025493399;0;-0.0910457025493399;0;0;0;-0.0910457025493399;0;0;0;-0.0910457025493399;0;0;0;-0.0910457025493399;0;-0.0910457025493399;0;0;0;0;0;0;0;-0.0910457025493399;0;0;0;-0.0910457025493399;0;0;0;0;0;0;0;-0.273137107648020;0;-0.273137107648020;0;-0.0910457025493399;0;0;0;-0.273137107648020;0;0;0;-0.273137107648020;0;-0.0910457025493399;0;-0.0910457025493399;0;0;0;0;0;-0.0910457025493399;0;0;0;-0.273137107648020;0;0;0;-0.273137107648020;0;0;0;0;0;-0.273137107648020;0;-0.273137107648020;0];
                Y4_ijkl_20 = [-0.315391565252520;0;0;0;-0.105130521750840;0;0;0;0.0525652608754200;0;-0.105130521750840;0;-0.105130521750840;0;0;0;0;0;0;0;0.0525652608754200;0;0;0;0.0525652608754200;0;0;0;-0.105130521750840;0;-0.105130521750840;0;0;0;0;0;-0.105130521750840;0;0;0;-0.315391565252520;0;0;0;0.0525652608754200;0;0;0;0;0;0.0525652608754200;0;0.0525652608754200;0;0;0;0.0525652608754200;0;0;0;0.0525652608754200;0;0;0;0;0;0;0;0.0525652608754200;0;0.0525652608754200;0;0.0525652608754200;0;0;0;0.0525652608754200;0;0;0;0.630783130505040];
                Y4_ijkl_21 = [0;0;-0.273137107648020;0;0;0;-0.273137107648020;0;0;0;0;0;0;0;-0.0910457025493399;0;-0.0910457025493399;0;-0.273137107648020;0;0;0;-0.0910457025493399;0;0;0;-0.273137107648020;0;0;0;0;0;-0.0910457025493399;0;-0.0910457025493399;0;0;0;-0.0910457025493399;0;0;0;-0.0910457025493399;0;0;0;-0.0910457025493399;0;-0.0910457025493399;0;0;0;0;0;-0.273137107648020;0;0;0;-0.0910457025493399;0;0;0;-0.273137107648020;0;-0.0910457025493399;0;-0.0910457025493399;0;0;0;0;0;0;0;-0.273137107648020;0;0;0;-0.273137107648020;0;0];
                Y4_ijkl_22 = [0.546274215296039;0;0;0;0;0;0;0;0.0910457025493399;0;0;0;0;0;0;0;0;0;0;0;0.0910457025493399;0;0;0;0.0910457025493399;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;-0.546274215296039;0;0;0;-0.0910457025493399;0;0;0;0;0;-0.0910457025493399;0;-0.0910457025493399;0;0;0;0.0910457025493399;0;0;0;0.0910457025493399;0;0;0;0;0;0;0;-0.0910457025493399;0;-0.0910457025493399;0;0.0910457025493399;0;0;0;-0.0910457025493399;0;0;0;0];
                % Y_{ijkl}^{4m}
                Y4_ijkl_4m4= [0;0.625835735449177;0;0.625835735449177;0;0;0;0;0;0.625835735449177;0;0;0;-0.625835735449177;0;0;0;0;0;0;0;0;0;0;0;0;0;0.625835735449177;0;0;0;-0.625835735449177;0;0;0;0;0;-0.625835735449177;0;-0.625835735449177;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0];
                Y4_ijkl_4m3= [0;0;0;0;0;-0.442532692444983;0;-0.442532692444983;0;0;0;-0.442532692444983;0;0;0;-0.442532692444983;0;0;0;-0.442532692444983;0;-0.442532692444983;0;0;0;0;0;0;0;-0.442532692444983;0;0;0;-0.442532692444983;0;0;0;0;0;0;0;0.442532692444983;0;0.442532692444983;0;-0.442532692444983;0;0;0;0.442532692444983;0;0;0;0;0;-0.442532692444983;0;-0.442532692444983;0;0;0;0;0;-0.442532692444983;0;0;0;0.442532692444983;0;0;0;0;0;0;0;0;0;0;0;0;0];
                Y4_ijkl_4m2= [0;-0.236543673939390;0;-0.236543673939390;0;0;0;0;0;-0.236543673939390;0;0;0;-0.236543673939390;0;0;0;0.473087347878780;0;0;0;0;0;0.473087347878780;0;0.473087347878780;0;-0.236543673939390;0;0;0;-0.236543673939390;0;0;0;0.473087347878780;0;-0.236543673939390;0;-0.236543673939390;0;0;0;0;0;0;0;0.473087347878780;0;0;0;0.473087347878780;0;0;0;0;0;0;0;0.473087347878780;0;0.473087347878780;0;0;0;0.473087347878780;0;0;0;0.473087347878780;0;0;0;0.473087347878780;0;0.473087347878780;0;0;0;0;0];
                Y4_ijkl_4m1= [0;0;0;0;0;0.167261635889322;0;0.167261635889322;0;0;0;0.167261635889322;0;0;0;0.167261635889322;0;0;0;0.167261635889322;0;0.167261635889322;0;0;0;0;0;0;0;0.167261635889322;0;0;0;0.167261635889322;0;0;0;0;0;0;0;0.501784907667967;0;0.501784907667967;0;0.167261635889322;0;0;0;0.501784907667967;0;0;0;-0.669046543557289;0;0.167261635889322;0;0.167261635889322;0;0;0;0;0;0.167261635889322;0;0;0;0.501784907667967;0;0;0;-0.669046543557289;0;0;0;0;0;-0.669046543557289;0;-0.669046543557289;0];
                Y4_ijkl_40 = [0.317356640745613;0;0;0;0.105785546915204;0;0;0;-0.423142187660817;0;0.105785546915204;0;0.105785546915204;0;0;0;0;0;0;0;-0.423142187660817;0;0;0;-0.423142187660817;0;0;0;0.105785546915204;0;0.105785546915204;0;0;0;0;0;0.105785546915204;0;0;0;0.317356640745613;0;0;0;-0.423142187660817;0;0;0;0;0;-0.423142187660817;0;-0.423142187660817;0;0;0;-0.423142187660817;0;0;0;-0.423142187660817;0;0;0;0;0;0;0;-0.423142187660817;0;-0.423142187660817;0;-0.423142187660817;0;0;0;-0.423142187660817;0;0;0;0.846284375321635];
                Y4_ijkl_41 = [0;0;0.501784907667967;0;0;0;0.501784907667967;0;0;0;0;0;0;0;0.167261635889322;0;0.167261635889322;0;0.501784907667967;0;0;0;0.167261635889322;0;0;0;-0.669046543557289;0;0;0;0;0;0.167261635889322;0;0.167261635889322;0;0;0;0.167261635889322;0;0;0;0.167261635889322;0;0;0;0.167261635889322;0;0.167261635889322;0;0;0;0;0;0.501784907667967;0;0;0;0.167261635889322;0;0;0;-0.669046543557289;0;0.167261635889322;0;0.167261635889322;0;0;0;0;0;0;0;-0.669046543557289;0;0;0;-0.669046543557289;0;0];
                Y4_ijkl_42 = [-0.473087347878780;0;0;0;0;0;0;0;0.473087347878780;0;0;0;0;0;0;0;0;0;0;0;0.473087347878780;0;0;0;0.473087347878780;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0.473087347878780;0;0;0;-0.473087347878780;0;0;0;0;0;-0.473087347878780;0;-0.473087347878780;0;0;0;0.473087347878780;0;0;0;0.473087347878780;0;0;0;0;0;0;0;-0.473087347878780;0;-0.473087347878780;0;0.473087347878780;0;0;0;-0.473087347878780;0;0;0;0];
                Y4_ijkl_43 = [0;0;-0.442532692444983;0;0;0;-0.442532692444983;0;0;0;0;0;0;0;0.442532692444983;0;0.442532692444983;0;-0.442532692444983;0;0;0;0.442532692444983;0;0;0;0;0;0;0;0;0;0.442532692444983;0;0.442532692444983;0;0;0;0.442532692444983;0;0;0;0.442532692444983;0;0;0;0.442532692444983;0;0.442532692444983;0;0;0;0;0;-0.442532692444983;0;0;0;0.442532692444983;0;0;0;0;0;0.442532692444983;0;0.442532692444983;0;0;0;0;0;0;0;0;0;0;0;0;0;0];
                Y4_ijkl_44 = [0.625835735449177;0;0;0;-0.625835735449177;0;0;0;0;0;-0.625835735449177;0;-0.625835735449177;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;-0.625835735449177;0;-0.625835735449177;0;0;0;0;0;-0.625835735449177;0;0;0;0.625835735449177;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0];
                if CSphase
                    Y4_ijkl_2m1=-Y4_ijkl_2m1;
                    Y4_ijkl_21=-Y4_ijkl_21;
                    Y4_ijkl_4m3=-Y4_ijkl_4m3;
                    Y4_ijkl_4m1=-Y4_ijkl_4m1;
                    Y4_ijkl_41=-Y4_ijkl_41;
                    Y4_ijkl_43=-Y4_ijkl_43;
                end
                Y_ell=[Y4_ijkl_00, Y4_ijkl_2m2, Y4_ijkl_2m1, Y4_ijkl_20, Y4_ijkl_21, Y4_ijkl_22, Y4_ijkl_4m4, Y4_ijkl_4m3, Y4_ijkl_4m2, Y4_ijkl_4m1, Y4_ijkl_40, Y4_ijkl_41, Y4_ijkl_42, Y4_ijkl_43, Y4_ijkl_44];
            else
                error('Only Lmax = 2 or 4 are supported')
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
        function Ylm_n = evaluate_even_SH(dirs, Lmax, CS_phase)
            % Ylm_n = get_even_SH(dirs,Lmax,CS_phase)
            %
            % if CS_phase=1, then the definition uses the Condon-Shortley phase factor
            % of (-1)^m. Default is CS_phase=0 (so this factor is ommited)
            %
            % By: Santiago Coelho (10/06/2021)
            
            
            if size(dirs,2)~=3
                dirs=dirs';
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
            K_lm=sqrt((2*l_all+1)./(4*pi) .* factorial(l_all-abs(m_all))./factorial(l_all+abs(m_all)));
            if nargin==2 || isempty(CS_phase) || ~exist('CS_phase','var') || ~CS_phase
                extra_factor=ones(size(K_lm));
                extra_factor(m_all~=0)=sqrt(2);
            else
                extra_factor=ones(size(K_lm));
                extra_factor(m_all~=0)=sqrt(2);
                extra_factor=extra_factor.*(-1).^(m_all);
            % % % % % %     extra_factor(m_all>0)=1;
            end
            P_l_in_cos_theta=zeros(length(l_all),Nmeas);
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
            Y_lm=repmat(extra_factor',1,Nmeas).*repmat(K_lm',1,Nmeas).*phi_term.*P_l_in_cos_theta;
            Ylm_n=Y_lm';
        end
        % =================================================================
        function DKI_maps = get_DKI_fiberBasis_maps_from_4D_DW_tensors(dt, mask, CSphase)
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
            end
            dt=double(dt);
            dt(~isfinite(dt(:)))=0;
            C0=sqrt(1/(4*pi));
            C2=sqrt(5/(4*pi));
            C4=sqrt(9/(4*pi));   

            % Transform Slm into Wlm (they are proportional)
            md=C0*abs(dt(:,:,:,1));
            dt_stf=RICE.vectorize(cat(4,dt(:,:,:,1:6),3*dt(:,:,:,7:21)./md.^2),mask);
            MDSq=(C0*dt_stf(1,:)).^2;  

            % Compute Diffusion and Kurtosis rotational invariants
            D00=dt(:,:,:,1);
            D0=C0*abs(D00);
            D2m=dt(:,:,:,2:6);
            D2=C2*sqrt(sum(D2m.^2,4));
            S00=dt(:,:,:,7);
            S0=C0*abs(S00);
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
            dirs256 = RICE.get256dirs();
            Y_LM_matrix256 = RICE.evaluate_even_SH(dirs256,4,CSphase);
            adc = Y_LM_matrix256(:,1:6)*dt_stf(1:6,:);
            akc = (Y_LM_matrix256*dt_stf(7:21,:)).*MDSq./(adc.^2);
            DKI_maps.mk=RICE.vectorize(mean(akc,1),mask);

            % Computing AW and RW with their exact definitions an all K maps
            Y00=eye(3)/sqrt(4*pi);
            Y2m2=[0,0.546274215296040,0;0.546274215296040,0,0;0,0,0];
            Y2m1=(-1)^(CSphase)*[0,0,0;0,0,-0.546274215296040;0,-0.546274215296040,0];
            Y20= [-0.315391565252520,0,0;0,-0.315391565252520,0;0,0,0.630783130505040];
            Y21= (-1)^(CSphase)*[0,0,-0.546274215296040;0,0,0;-0.546274215296040,0,0];
            Y22= [0.546274215296040,0,0;0,-0.546274215296040,0;0,0,0];
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
                [eigvec, eigval] = eigs(DT);
                e1(:, ii) = eigvec(:, 1);
                lambda(:,ii) = diag(eigval);
                % Getting AW(AK), RW, and RK
                Ylm_e1 = RICE.evaluate_even_SH(eigvec(:, 1),4,CSphase);
                wpar(ii)=Ylm_e1*Wlm;
                dirs_radial = RICE.radialsampling(eigvec(:, 1), 100);
                Ylm_radial = RICE.evaluate_even_SH(dirs_radial,4,CSphase);
                D_n_radial = Ylm_radial(:,1:6)*Dlm;
                W_n_radial=Ylm_radial*Wlm;
                wperp(ii)=mean(W_n_radial,1);
        %         rk(ii)=mean(Ylm_radial(:,1:15)*wlm(1:15,ii)*MDSq(ii)/((l2(ii)/2+l3(ii)/2)^2));
                rk(ii)=MDSq(ii)*mean(W_n_radial./D_n_radial.^2,1);
            end
            ad = RICE.vectorize(lambda(1,:), mask);
            rd = RICE.vectorize(sum(lambda(2:3,:),1)/2, mask);
            wpar = RICE.vectorize(wpar, mask);
            wperp = RICE.vectorize(wperp, mask);
            DKI_maps.rd=rd;
            DKI_maps.ad=ad;
            DKI_maps.rw=wperp.*D0.^2./rd.^2;
            DKI_maps.aw=wpar.*D0.^2./ad.^2;
            DKI_maps.rk=RICE.vectorize(rk,mask);
            DKI_maps.ak=DKI_maps.aw; % They are the same
            DKI_maps.fe=RICE.vectorize(e1,mask);
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
    end
end

