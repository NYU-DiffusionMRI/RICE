function [AUC_logreg, coeffs, output_pred, output_gt] = RICE_logistic_regression_AUC_stratified(X_input,y_input,Nboot,stratify)
% [AUC_logreg, coeffs] = RICE_logistic_regression_AUC_stratified(X_input,y_input,Nboot,stratify)

    if ~exist('Nboot', 'var') || isempty(Nboot)
        Nboot = 1;
    end

    coeffs = zeros(size(X_input,2)+1,Nboot);
    AUC_logreg = zeros(Nboot,1);
    for id_bootstrap = 1:Nboot
        % Split into training and test sets
        idx_train = balanced_age_matched_subsample_local(stratify.sex, stratify.ms, stratify.age, stratify.train);
        idx_test = ~idx_train;
        Xtrain = X_input(idx_train, :);
        ytrain = y_input(idx_train);
        Xtest  = X_input(idx_test, :);
        ytest  = y_input(idx_test);
        % Train logistic regression
        logregModel = fitglm(Xtrain, ytrain, 'Distribution', 'binomial', 'Link', 'logit');
        coeffs(:,id_bootstrap) = logregModel.Coefficients.Estimate;
        % Predict probabilities for test data
        prob_logreg = predict(logregModel, Xtest);
        % Compute AUC
        [~,~,~,AUC_logreg(id_bootstrap)] = perfcurve(ytest, prob_logreg, 1);
        output_pred{id_bootstrap} = prob_logreg;
        output_gt{id_bootstrap} = ytest;
    end
end

function idx = balanced_age_matched_subsample_local(sex, group, age, frac)
% % BALANCED_AGE_MATCHED_SAMPLE
% %   Selects a random subset of subjects preserving the original
% %   MS/control proportion and age distribution (age-matched).
% %
% % INPUTS
% %   sex   - [N x 1] vector, 0/1 for sex
% %   group - [N x 1] vector, 0=control, 1=MS
% %   age   - [N x 1] vector with ages
% %   frac  - fraction of subjects to sample (e.g., 0.8 for 80%)
% %
% % OUTPUT
% %   idx   - logical index vector, 1 = subject selected
% 
%     if nargin < 4
%         frac = 0.8;
%     end
% 
%     N = numel(group);
%     Nsample = round(frac * N);
% 
%     % global MS proportion
%     pMS = mean(group == 1);
%     nMS_target = round(pMS * Nsample);
%     nCT_target = Nsample - nMS_target;
% 
%     % Define age bins (5-year bins, can adjust)
%     edges = min(age):5:max(age);
%     ageBins = discretize(age, edges);
% 
%     idx = false(N,1);
%     selectedMS = 0; selectedCT = 0;
% 
%     % Loop through bins
%     for b = 1:max(ageBins)
%         binIdx = (ageBins == b);
%         MS_in_bin = find(binIdx & group == 1);
%         CT_in_bin = find(binIdx & group == 0);
% 
%         if isempty(MS_in_bin) && isempty(CT_in_bin)
%             continue;
%         end
% 
%         % Desired proportion in this bin (same as global)
%         nInBin = round(frac * sum(binIdx));
%         nMS_bin = round(pMS * nInBin);
%         nCT_bin = nInBin - nMS_bin;
% 
%         % Sample with replacement if fewer available
%         if numel(MS_in_bin) < nMS_bin
%             chosenMS = randsample(MS_in_bin, nMS_bin, true);
%         else
%             chosenMS = randsample(MS_in_bin, nMS_bin, false);
%         end
% 
%         if numel(CT_in_bin) < nCT_bin
%             chosenCT = randsample(CT_in_bin, nCT_bin, true);
%         else
%             chosenCT = randsample(CT_in_bin, nCT_bin, false);
%         end
% 
%         idx([chosenMS; chosenCT]) = true;
%         selectedMS = selectedMS + numel(chosenMS);
%         selectedCT = selectedCT + numel(chosenCT);
%     end
% 
%     % Adjust if off by a few due to rounding
%     selectedTotal = selectedMS + selectedCT;
%     if selectedTotal > Nsample
%         surplus = find(idx);
%         drop = randsample(surplus, selectedTotal - Nsample);
%         idx(drop) = false;
%     elseif selectedTotal < Nsample
%         remaining = find(~idx);
%         add = randsample(remaining, Nsample - selectedTotal);
%         idx(add) = true;
%     end
% end


% BALANCED_AGE_SEX_SAMPLE
%   Selects a random subset of subjects preserving:
%   - Same MS/control proportion as original dataset
%   - Age-matched distributions
%   - Sex balance within each group
%
% INPUTS
%   sex   - [N x 1] vector, 0/1 for sex
%   group - [N x 1] vector, 0=control, 1=MS
%   age   - [N x 1] vector with ages
%   frac  - fraction of subjects to sample (e.g., 0.8 for 80%)
%
% OUTPUT
%   idx   - logical index vector, 1 = subject selected

    if nargin < 4
        frac = 0.8;
    end
    
    N = numel(group);
    Nsample = round(frac * N);
    
    % Global proportions
    pMS  = mean(group == 1);
    pSex = mean(sex == 1);  % not enforced globally, but per group
    
    % Target counts
    nMS_target = round(pMS * Nsample);
    nCT_target = Nsample - nMS_target;
    
    % Age bins (5-year bins, adjust as needed)
    edges = min(age):5:max(age);
    if numel(edges) == 1  % all ages the same
        edges = [min(age)-1, max(age)+1];
    end
    ageBins = discretize(age, edges);
    
    idx = false(N,1);
    selectedCount = 0;
    
    % Loop over age bins
    for b = 1:max(ageBins)
        binIdx = (ageBins == b);
        if ~any(binIdx), continue; end
        
        % Subjects in this bin
        MS_in_bin = find(binIdx & group == 1);
        CT_in_bin = find(binIdx & group == 0);
        
        % Desired number in this bin (proportional to bin size)
        nInBin = round(frac * sum(binIdx));
        if nInBin == 0, continue; end
        
        nMS_bin = round(pMS * nInBin);
        nCT_bin = nInBin - nMS_bin;
        
        % --- MS group, stratify by sex ---
        for s = 0:1
            idxMSsex = MS_in_bin(sex(MS_in_bin) == s);
            if isempty(idxMSsex), continue; end
            % desired per-sex proportion in MS (relative to MS in bin)
            pSexMS = mean(sex(MS_in_bin) == s);
            nTarget = round(pSexMS * nMS_bin);
            if numel(idxMSsex) < nTarget
                chosen = randsample(idxMSsex, nTarget, true);
            else
                chosen = randsample(idxMSsex, nTarget, false);
            end
            idx(chosen) = true;
            selectedCount = selectedCount + numel(chosen);
        end
        
        % --- Control group, stratify by sex ---
        for s = 0:1
            idxCTsex = CT_in_bin(sex(CT_in_bin) == s);
            if isempty(idxCTsex), continue; end
            % desired per-sex proportion in controls (relative to controls in bin)
            pSexCT = mean(sex(CT_in_bin) == s);
            nTarget = round(pSexCT * nCT_bin);
            if numel(idxCTsex) < nTarget
                chosen = randsample(idxCTsex, nTarget, true);
            else
                chosen = randsample(idxCTsex, nTarget, false);
            end
            idx(chosen) = true;
            selectedCount = selectedCount + numel(chosen);
        end
    end
    
    % Adjust if mismatch due to rounding
    if selectedCount > Nsample
        drop = randsample(find(idx), selectedCount - Nsample);
        idx(drop) = false;
    elseif selectedCount < Nsample
        add = randsample(find(~idx), Nsample - selectedCount);
        idx(add) = true;
    end

    % Adjust if mismatch due to rounding
    if selectedCount > Nsample
        drop = randsample(find(idx), selectedCount - Nsample);
        idx(drop) = false;
    elseif selectedCount < Nsample
        add = randsample(find(~idx), Nsample - selectedCount);
        idx(add) = true;
    end

end
