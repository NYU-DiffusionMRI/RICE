%% Analyzing RICE summary statistics

% Recomputing bootstrapped AUC values from RICE summary statistics (~2.5 minutes in a normal desktop)

clc,clear,close all

% Load .mat file with summary statistics
root = '/Users/coelhs01/Documents/SantiagoCoelho/NYU_Postdoc_MyScience/MATLAB/Projects/RICE_B/';
% root = '/Users/coelhs01/Documents/SantiagoCoelho/Git/RICE';
load(fullfile(root,'RICE_MS_summary_statistics.mat'))

rng default 

subsets_all{1} = [1 2 3];
subset_tags{1,1} = 'DTI'; % DTI invariants
subsets_all{2} = [1 2 3 4 5 7];
subset_tags{2,1} = 'DKI'; % DKI invariants
subsets_all{3} = 1:15;
subset_tags{3,1} = ' RICE$_\mathrm{LTE}$'; % RICE intrinsic+mixed invariants

Nboot = 100;
tic
AUC_logReg = zeros(Nr, length(subsets_all),Nboot);
for id_subset = 1:length(subsets_all)
    subset_idx = subsets_all{id_subset};
    for id_roi = 1:Nr
        X_keep = squeeze(rice_median(subset_idx,id_roi,:))';
        X_keep_agesex = [ X_keep age_all(:) sex_all(:) ];
        stratify.sex = sex_all(:); stratify.ms = flag_ms(:); stratify.age = age_all(:); stratify.train = 0.8;
        % [AUC_logReg(id_roi,id_subset,:),current_coeffs, output_pred(id_roi,id_subset,:,:)] = compute_logistic_regression_AUC_stratified(X_keep_agesex,flag_ms(:),Nboot,stratify);
        [AUC_logReg(id_roi,id_subset,:),current_coeffs, output_pred(id_roi,id_subset,:,:)] = RICE_logistic_regression_AUC_stratified(X_keep_agesex,flag_ms(:),Nboot,stratify);
        coeffs_mean{id_roi,id_subset} = mean(current_coeffs,2);
        coeffs_std{id_roi,id_subset} = std(current_coeffs,[],2);
    end
end
t = toc; fprintf('Time for %d AUC computations = %.4f seconds \n',Nboot,t)

%% Plot the above AUCs (combined features + age + sex)
clc,close all

% rois_keep = [ 1:19 ]; % All delateralized ROIs
rois_keep = [ 1 3 10 11 12 13 14 19 ]; % Larger ROIs + involved in MS

subsets_keep = [1 2 3]; % Which sets of invariants are plotted

% Create bar plot with 95% confidence intervals
ROI_name_abbrev = {'GCC','BCC','SCC','CST','ML','ICP','SCP','CP','ALIC','PLIC','ACR','SCR','PCR','PTR','C','SLF','T','EC','TWM'};
names = ROI_name_abbrev(rois_keep);

% Mean and standard deviation across bootstrap resamples
AUC_means = mean(AUC_logReg(rois_keep,subsets_keep,:), 3); 
AUC_std   = std(AUC_logReg(rois_keep,subsets_keep,:), 0, 3);

% 95% CI for the mean AUC across bootstrap resamples (normal approximation)
CI_95 = 1.96 * AUC_std ./ sqrt(Nboot);


% Grouped bar plot
figure('Position',[2963 590 1024 629]), hold on
h = bar(AUC_means,'grouped');
% Set colors
blue_shade = [ 167 202 236 ; 78 149 217 ; 33 95 154 ]/256;
for k = 1:length(subsets_keep)
    h(k).FaceColor = blue_shade(k,:);
end

% Add errorbars
numGroups = size(AUC_means,1);
numBars   = size(AUC_means,2);

% X positions of the bars
groupWidth = min(0.8, numBars/(numBars+1.5));  
for i = 1:numBars
    % Get center of each bar
    x = (1:numGroups) - groupWidth/2 + (2*i-1) * groupWidth / (2*numBars);
    errorbar(x, AUC_means(:,i), CI_95(:,i), 'k', 'linestyle', 'none', 'LineWidth',1.5);
end

% Adjust x-axis labels: one label per subject
set(gca,'XTick',1:length(rois_keep),'XTickLabel',names);
set(gca,'TickLength',[0 0])    % removes the actual tick marks
ax = gca;                 % get current axes
ax.TitleFontSizeMultiplier = 1.5;   % scales relative to default

% Legend
legend(subset_tags(subsets_keep),'Location','northwest','interpreter','latex');
title('MS classification AUC - logistic regression','interpreter','latex');
set(gca,'FontSize',15)
set(gca, 'TickLabelInterpreter', 'latex');
ylim([0.6 0.95]), box on
set(gca, 'LineWidth', 0.8, 'XColor', 'k', 'YColor', 'k')
