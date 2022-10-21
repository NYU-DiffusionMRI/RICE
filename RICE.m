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
        function [b0, tensor_elems, RICE_maps, DIFF_maps] = fit(DWI, b, dirs, bshape, mask, CSphase, type, nls_flag)
            % [b0, tensor_elems, RICE_maps, DIFF_maps] = fit(DWI, b, dirs, bshape, mask, CSphase, type, nls_flag)
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
            %       type can be 'minimalDKI' and then tensor_elem has:   [D00 D2m S00] (7 elem)
            %                                      RICE has: D0 D2 S0
            %                                      maps has: md fa mw
            %       type can be 'minimalDKI_iso' and then tensor_elem has:   [D00 S00] (2 elem)
            %                                      RICE has: D0 S0
            %                                      maps has: md mw 
            %       type can be 'fullDKI' and then tensor_elem has:   [D00 D2m S00 S2m S4m] (21 elem)
            %                                      RICE has: D0 D2 S0 S2 S4
            %                                      maps has: md fa mw rw aw (these are axial symmetry approximations) 
            %       type can be 'minimalRICE' and then tensor_elem has:  [D00 D2m S00 S2m A00(rank2)] (14 elem)
            %                                      RICE has: D0 D2 S0 S2 A0
            %                                      maps has: md ad rd fa mw ufa 
            %       type can be 'fullRICE' and then tensor_elem has:  [D00 D2m S00 S2m S4m A00(rank2) A2m(rank2)] (27 elem)
            %                                      RICE has: D0 D2 S0 S2 S4 A0 A2
            %                                      maps has: md ad rd fa mw aw rw ufa d0d2m (these are axial symmetry approximations) 
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
            elseif strcmp(type,'minimalDKI_iso')
                r=2; % 2 elements are fitted: D00 S00
                X=[ones(size(b)), -Bij*Y2(:,1), 1/2*BijBkl*Y4(:,1)];
                flag_S0=1;
            elseif strcmp(type,'minimalDKI')
                r=12; % 7 elements are fitted: D00 D2m S00 (S2m are fitted and discarded)
                X=[ones(size(b)), -Bij*Y2, 1/2*BijBkl*Y4(:,1:6)];
                flag_S0=1; flag_D2=1;
            elseif strcmp(type,'fullDKI')
                r=21; % 21 elements are fitted: D00 D2m S00 S2m S4m
                X=[ones(size(b)), -Bij*Y2, 1/2*BijBkl*Y4];
                flag_S0=1; flag_D2=1; flag_S2=1; flag_S4=1;
            elseif strcmp(type,'minimalRICE')
                r=13; % 8 elements are fitted: D00 D2m S00 A00(rank2) (S2m are fitted and discarded)
                X=[ones(size(b)), -Bij*Y2, 1/2*BijBkl*Y4(:,1:6), 1/12*BijBkl_epsilon_term*Y2(:,1)];
                flag_S0=1; flag_D2=1; flag_S2=1; flag_A0=1;
            elseif strcmp(type,'fullRICE')
                r=27; % 27 elements are fitted: D00 D2m S00 S2m S4m A00(rank2) A2m(rank2)
                X=[ones(size(b)), -Bij*Y2, 1/2*BijBkl*Y4, 1/12*BijBkl_epsilon_term*Y2];
                flag_S0=1; flag_D2=1; flag_S2=1; flag_S4=1; flag_A0=1; flag_A2=1;
            else
                error('Choose an appropriate signal representation')
            end    
            if Brank<r, error('rank of protocol B-tensor set is inconsistent with the desired fit'), end
            
            % unconstrained LLS fit
            dv = X\log(DWI);
            w = exp(X*dv);
            % WLLS fit initialized with LLS
            parfor ii = 1:Nvoxels
                wi = diag(w(:,ii));
                logdwii = log(DWI(:,ii));
                dv(:,ii) = (wi*X)\(wi*logdwii);
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
            if strcmp(type,'minimalDKI_iso')
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
                maps.mw=W0;
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
                D0sq=D0.^2+5/9*S0+1/6*A0;
                D2sq=D2.^2+20/9*S0-5/6*A0;
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
                D0sq=D0.^2+5/9*S0+1/6*A0;
                D2sq=D2.^2+20/9*S0-5/6*A0;
                ufa_sq=3*D2sq./(4*D0.^2+2*D2sq);
                ufa_sq(ufa_sq(:)<0)=0;
                DIFF_maps.ufa=sqrt(ufa_sq);
                DIFF_maps.d0d2m=sqrt(sum((7/(6*C0)*S2m-2/(3*C0)*A2m+D00.*D2m).^2,4));
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
    end
end

