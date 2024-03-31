clear
close all

% load('E:\PHD\learning\research\AD_two_modal\result\advanced_analysis\200roi\baseline\cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.5_fa_npi7_method_pca_fmri90_CAcomp3_fold10\brain_transformed.mat')
% load('E:\PHD\learning\research\AD_two_modal\result\advanced_analysis\200roi\baseline\cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.5_fa_npi7_method_pca_fmri90_CAcomp3_fold10\npi_transformed.mat')
load('E:\PHD\learning\research\AD_two_modal\result\multi_run\advanced_analysis\100roi\baseline\cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.6_pca_npi7_method_none_fmri4950_CAcomp4_fold10\brain_transformed.mat')
load('E:\PHD\learning\research\AD_two_modal\result\multi_run\advanced_analysis\100roi\baseline\cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.6_pca_npi7_method_none_fmri4950_CAcomp4_fold10\npi_transformed.mat')

label = load('E:\PHD\learning\research\AD_two_modal\result\multi_run\advanced_analysis\100roi\baseline\cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.6_pca_npi7_method_none_fmri4950_CAcomp4_fold10\label_diagnosis.mat');
label = string(label.label)';
hc_mask = label == 'Cognitively normal';
% brain_feature = brain_feature(~hc_mask,:);
iteration = 2;
cluster_num = 3;

% brain_feature = 20 + (80-20) .* rand(177, 100);

brain_feature = normalize(brain_feature);
clustFcn = @(X,K) kmeans(X, K,'Replicates',5, 'Options',statset('UseParallel',1));
[idx,Center_true] = kmeans(brain_feature(:,1:2), cluster_num,'Replicates',5, 'Options',statset('UseParallel',1));
c_true = idx;


gap = [];
DaviesBouldin = [];
Silhouette = [];
CalinskiHarabasz = [];
labels = zeros(length(brain_feature), iteration);
centers = zeros(cluster_num, 2, iteration);
for i = 1:iteration
    i
    idx2 = randperm(length(brain_feature));
    mask = idx2(1:round(length(brain_feature)*0.9));
    evaluation1 = evalclusters(brain_feature(mask,1:2),clustFcn,"gap","KList",1:8);
    evaluation2 = evalclusters(brain_feature(mask,1:2),clustFcn,"DaviesBouldin","KList",2:8);
    evaluation3 = evalclusters(brain_feature(mask,1:2),clustFcn,"Silhouette","KList",2:8);
    evaluation4 = evalclusters(brain_feature(mask,1:2),clustFcn,"CalinskiHarabasz","KList",2:8);
    gap = [gap; evaluation1.CriterionValues];
    DaviesBouldin = [DaviesBouldin; evaluation2.CriterionValues];
    Silhouette = [Silhouette; evaluation3.CriterionValues];
    CalinskiHarabasz = [CalinskiHarabasz; evaluation4.CriterionValues];
    [idx,C] = kmeans(brain_feature(mask,1:2), cluster_num,'Replicates',1, 'Options',statset('UseParallel',1));
    labels(mask, i) = idx;
    centers(:, :, i) = C;
end

gap_mean = mean(gap);
gap_std = std(gap);
DaviesBouldin_mean = mean(DaviesBouldin);
DaviesBouldin_std = std(DaviesBouldin);
Silhouette_mean = mean(Silhouette);
Silhouette_std = std(Silhouette);
CalinskiHarabasz_mean = mean(CalinskiHarabasz);
CalinskiHarabasz_std = std(CalinskiHarabasz);

error_plot(1:8, gap_mean, gap_std, 'Gap value', 'Number of clusters', 0, 0.8, 0.5, 8.5);
error_plot(2:8, DaviesBouldin_mean, DaviesBouldin_std, 'Davies–Bouldin index',  'Number of clusters',0.7, 1.3, 1.5, 8.5);
error_plot(2:8, Silhouette_mean, Silhouette_std, 'Silhouette coefficient', 'Number of clusters', 0.35, 0.6, 1.5, 8.5);
error_plot(2:8, CalinskiHarabasz_mean, CalinskiHarabasz_std, 'Calinski-Harabasz criterion', 'Number of clusters', 40, 70, 1.5, 8.5);

certer_mean = mean(centers,3);
certer_mean1 = squeeze(certer_mean(1,:));
certer_mean2 = squeeze(certer_mean(2,:));
%%%%%%%%%%%%%%%%%%%%%%%label resort
labels_resort = zeros(length(brain_feature), iteration);
label_id_resort = zeros(cluster_num, iteration);
possible_ind = perms(1:cluster_num);
similarity_all = zeros(iteration, length(possible_ind));

for i =1:iteration
    for j = 1:cluster_num
        label_id_resort(j,i)=j;
    end
    label1_raw = squeeze(c_true);
    label2_raw = squeeze(labels(:,i));
    mask = label1_raw~=0 & label2_raw ~= 0;
    label1 = label1_raw(mask);
    label2 = label2_raw(mask);

    similarity = pdist2(label1',label2','jaccard');
    labels_resort(:,i) = label2_raw;
    for j = 1: length(possible_ind)
        label3_raw = squeeze(labels(:,i));
        label3_raw2 = squeeze(labels(:,i));
        label3 = label3_raw(mask);
        label3_ = label3_raw(mask);
        label_new = [];
        for k = 1: size(possible_ind,2)
            label3(label3_==k)=possible_ind(j,k);
            label3_raw(label3_raw2==k)=possible_ind(j,k);
            label_new=[label_new possible_ind(j,k)];
        end
        similarity1 = pdist2(label1',label3','jaccard');
        similarity_all(i,j) = similarity1;

        if similarity1 < similarity
            labels_resort(:,i) = label3_raw;
            similarity = similarity1;
            label_id_resort(:,i) = label_new;
        end
    end
end

certer_mean_resort = zeros(iteration, cluster_num, 2);

for i =1:iteration
    certer_mean_resort(i,:,:) = centers(label_id_resort(:,i),:,i);
end
%%%%%%%%%%%%%%%%%%%%%%%%%center plot
certer_mean_resort1 = squeeze(certer_mean_resort(:,1,:));
certer_mean_resort2 = squeeze(certer_mean_resort(:,2,:));
certer_mean_resort3 = squeeze(certer_mean_resort(:,3,:));
[val, idx] = sort(pdist2(certer_mean_resort3,[-0.28,1.22]));

% figure('units','normalized','outerposition',[.2 .2 .25 .5]);
% colormap(othercolor('Blues9'));
% scatplot(certer_mean_resort1(idx(1:700),1), certer_mean_resort1(idx(1:700),2));
% hold on; 
% colormap(othercolor('Reds9'));
% scatplot(certer_mean_resort2(idx(1:700),1), certer_mean_resort2(idx(1:700),2));
% hold on; 
% colormap(othercolor('Greens9'));
% scatplot(certer_mean_resort3(idx(1:700),1), certer_mean_resort3(idx(1:700),2));
% hold off; 
% xlabel('Mood variate') 
% ylabel('Anxiety variate') 
% set(gca,'FontSize',12, 'FontName', 'Tahoma', 'FontWeight','bold');
% minx = min(certer_mean_resort(:,:,1),[],'all');
% maxx = max(certer_mean_resort(:,:,1),[],'all');
% miny = min(certer_mean_resort(:,:,2),[],'all');
% maxy = max(certer_mean_resort(:,:,2),[],'all');
% unitx = round((maxx - minx)/4, 2);
% unity = round((maxy - miny)/4, 2);
% tickx = [minx minx+unitx minx+2*unitx minx+3*unitx minx+4*unitx];
% ticky = [miny miny+unity miny+2*unity miny+3*unity miny+4*unity];
% yticks(ticky);
% xticks(tickx);
% ytickformat('%.1f')
% xtickformat('%.1f')

label_stable1 = [];
label_stable2 = [];
label_stable3 = [];
label_stable4 = [];
label_stable5 = [];
label_stable6 = [];
label_stable7 = [];
label_stable8 = [];
for i = 1:length(brain_feature)
    labels_tem = labels_resort(i,labels_resort(i,:)~=0);
    if c_true(i) == 1
        label1_num = sum(labels_tem==1);
        num = label1_num;
        ratio = num/length(labels_tem);
        label_stable1 = [label_stable1 ratio];
    elseif c_true(i) == 2
        label2_num = sum(labels_tem==2);
        num = label2_num;
        ratio = num/length(labels_tem);
        label_stable2 = [label_stable2 ratio];
    elseif c_true(i) ==3
        label3_num = sum(labels_tem==3);
        num = label3_num;
        ratio = num/length(labels_tem);
        label_stable3 = [label_stable3 ratio];
    elseif c_true(i) ==4
        label4_num = sum(labels_tem==4);
        num = label4_num;
        ratio = num/length(labels_tem);
        label_stable4 = [label_stable4 ratio];
    elseif c_true(i) ==5
        label4_num = sum(labels_tem==5);
        num = label4_num;
        ratio = num/length(labels_tem);
        label_stable5 = [label_stable5 ratio];
    elseif c_true(i) ==6
        label6_num = sum(labels_tem==6);
        ratio = label6_num/length(labels_tem);
        label_stable6 = [label_stable6 ratio];
    elseif c_true(i) ==7
        label7_num = sum(labels_tem==7);
        ratio = label7_num/length(labels_tem);
        label_stable7 = [label_stable7 ratio];
    elseif c_true(i) ==8
        label7_num = sum(labels_tem==8);
        ratio = label7_num/length(labels_tem);
        label_stable8 = [label_stable8 ratio];
    end
end

all_stable = [label_stable1 label_stable2 label_stable3 label_stable4 label_stable5 label_stable6 label_stable7 label_stable8];
% all_stable = [label_stable1 label_stable2];
all_stable_mean = mean(all_stable);
all_stable_std = std(all_stable);
label_stable1_mean = mean(label_stable1);
label_stable1_std = std(label_stable1);
label_stable2_mean = mean(label_stable2);
label_stable2_std = std(label_stable2);
label_stable3_mean = mean(label_stable3);
label_stable3_std = std(label_stable3);

x = [1 3 4 5];
vals = [all_stable_mean; label_stable1_mean; label_stable2_mean; label_stable3_mean];

figure('units','normalized','outerposition',[.2 .2 .25 .5]);
b = bar(x,vals, 0.5, 'FaceColor','flat');
b(1).CData(1,:) = [0.5 0.5 0.5];
b(1).CData(2,:) = [1, 0.702, 0.702];  
b(1).CData(3,:) = [0.651, 0.8706, 0.9647];  
b(1).CData(4,:) = [0.4902, 0.6863, 0.223];  
b(1).EdgeColor = [1 1 1];

box off;
yticks([0.2 0.4 0.6 0.8 1.0])
xticklabels({'All', 'Subtype 1', 'Subtype 2', 'Subtype 3'});
set(gca,'FontSize',12, 'FontName', 'Tahoma', 'FontWeight','bold');
ylabel('Cluster stabilitiy') 
hold on
errhigh = [all_stable_std/2; label_stable1_std/2; label_stable2_std/2; label_stable3_std/2];
errlow = [all_stable_std/2; label_stable1_std/2; label_stable2_std/2; label_stable3_std/2];
er = errorbar(x,vals,errlow,errhigh);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
hold off
% 
% save(['H:\PHD\learning\research\AD_two_modal\result\final_figure\normalize\kmeans\stability\only_ad\k=', num2str(cluster_num), '.mat'], 'centers', 'c_true',...
%     'labels_resort', 'certer_mean_resort', 'all_stable');

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k2 = load ('H:\PHD\learning\research\AD_two_modal2\result\final_figure\normalize\kmeans\stability\k=2.mat');
k3 = load ('H:\PHD\learning\research\AD_two_modal2\result\final_figure\normalize\kmeans\stability\k=3.mat');
k4 = load ('H:\PHD\learning\research\AD_two_modal2\result\final_figure\normalize\kmeans\stability\k=4.mat');
k5 = load ('H:\PHD\learning\research\AD_two_modal2\result\final_figure\normalize\kmeans\stability\k=5.mat');
k6 = load ('H:\PHD\learning\research\AD_two_modal2\result\final_figure\normalize\kmeans\stability\k=6.mat');
k7 = load ('H:\PHD\learning\research\AD_two_modal2\result\final_figure\normalize\kmeans\stability\k=7.mat');
k8 = load ('H:\PHD\learning\research\AD_two_modal2\result\final_figure\normalize\kmeans\stability\k=8.mat');
stable_mean = [mean(k2.all_stable) mean(k3.all_stable) mean(k4.all_stable) mean(k5.all_stable) mean(k6.all_stable) mean(k7.all_stable) mean(k8.all_stable)];
stable_std = [std(k2.all_stable)/2 std(k3.all_stable)/2 std(k4.all_stable)/2 std(k5.all_stable)/2 std(k6.all_stable)/2 std(k7.all_stable)/2 std(k8.all_stable)/2];
error_plot(2:8, stable_mean, stable_std, 'Cluster stabilitiy', 'Number of clusters', 0.4, 1.3, 1.5, 8.5);
%     p = (sum(r>-0.18)+1)/1001

ratio = [];
for i = 1:length(k3.c_true)
    mask = k3.labels_resort(i,:)~=0;
    ratio = [ratio sum(k3.labels_resort(i,mask)==k3.c_true(i))/sum(mask)];
end
k3.c_true(ratio<0.95)=4;

figure('units','normalized','outerposition',[.2 .2 .25 .5]);
box off;
scatter(brain_feature(k3.c_true==1&hc_mask,1), brain_feature(k3.c_true==1&hc_mask,2), 20, 'filled','d', 'MarkerEdgeColor', 'white', ...
    'MarkerFaceColor', [1, 0.702, 0.702]);
hold on; 
scatter(brain_feature(k3.c_true==1&~hc_mask,1), brain_feature(k3.c_true==1&~hc_mask,2), 30, 'MarkerEdgeColor', 'white', ...
    'MarkerFaceColor', [1, 0.196, 0.196]);
hold on;
scatter(brain_feature(k3.c_true==2&hc_mask,1), brain_feature(k3.c_true==2&hc_mask,2), 20, 'filled','d', 'MarkerEdgeColor', 'white', ...
    'MarkerFaceColor', [0.651, 0.8706, 0.9647]);
hold on; 
scatter(brain_feature(k3.c_true==2&~hc_mask,1), brain_feature(k3.c_true==2&~hc_mask,2), 30, 'MarkerEdgeColor', 'white', ...
    'MarkerFaceColor', [0.192, 0.7333, 0.9647]);
hold on; 
scatter(brain_feature(k3.c_true==3&hc_mask,1), brain_feature(k3.c_true==3&hc_mask,2), 20, 'filled','d', 'MarkerEdgeColor', 'white', ...
    'MarkerFaceColor', [0.4902, 0.6863, 0.223]);
hold on; 
scatter(brain_feature(k3.c_true==3&~hc_mask,1), brain_feature(k3.c_true==3&~hc_mask,2), 30, 'MarkerEdgeColor', 'white', ...
    'MarkerFaceColor', [0.2471, 0.4353, 0.]);
hold on; 
scatter(brain_feature(k3.c_true==4&hc_mask,1), brain_feature(k3.c_true==4&hc_mask,2), 20, 'MarkerEdgeColor', 'white', ...
    'MarkerFaceColor', [.7 .7 .7]);
hold on; 
scatter(brain_feature(k3.c_true==4&~hc_mask,1), brain_feature(k3.c_true==4&~hc_mask,2), 20, 'filled','d', 'MarkerEdgeColor', 'white', ...
    'MarkerFaceColor', [.7 .7 .7]);
xlabel('Affective variate') 
ylabel('Anxiety variate') 
set(gca,'FontSize',12, 'FontName', 'Tahoma', 'FontWeight','bold');
minx = min(brain_feature(:,1),[],'all');
maxx = max(brain_feature(:,1),[],'all');
miny = min(brain_feature(:,2),[],'all');
maxy = max(brain_feature(:,2),[],'all');
unitx = round((maxx - minx)/4, 2);
unity = round((maxy - miny)/4, 2);
tickx = [minx minx+unitx minx+2*unitx minx+3*unitx minx+4*unitx];
ticky = [miny miny+unity miny+2*unity miny+3*unity miny+4*unity];
xticks(tickx);
yticks(ticky);
ytickformat('%.1f')
xtickformat('%.1f')
hold on; 
% x = [-1.63,0.78,0.667,0.424,-0.523,-2.46,-2.51,-2.30,-1.628];
% y = [2.88,2.83,1.31,0.848,0.114,-0.644,-0.586,0.946,2.88];
% fill(x, y, [1, 0.9, 0.9], 'FaceAlpha', 0.2);
% hold on; 
% plot(x, y, 'Color', [1, 0.3, 0.3], 'LineWidth',1);
% hold on; 
x = [1.056,1.43,2.129,1.587,-0.854,-0.789,-0.433,0.283,1.056];
y = [0.563,0.516,-0.626,-1.744,-1.894,-1.588,-0.806,0.279,0.563];
hold on; 
x = [-1.72,-1.08,-0.21,-0.09,-0.14,-0.25,-0.85,-2.46,-2.51,-1.90,-1.72];
y = [0.43,0.33,0.15,-0.016,-0.392,-0.700,-1.894,-0.644,-0.586,0.250,0.43];
fill(x, y, [0.651, 0.8706, 0.9647], 'FaceAlpha', 0.2);
hold on; 
plot(x, y, 'Color', [0.651, 0.8706, 0.9647], 'LineWidth',1);
hold on; 
x = [1.056,1.43,2.129,1.587,0.77,0.283,0.124,0.294,1.056];
y = [0.563,0.516,-0.626,-1.744,-1.764,-1.491,-0.24,0.20,0.56];
fill(x, y, [1, 0.702, 0.702], 'FaceAlpha', 0.2);
hold on; 
plot(x, y, 'Color', [1, 0.702, 0.702], 'LineWidth',1);
hold on; 
x = [-1.63,0.78,1.13,0.727,0.483,-0.13,-0.54,-0.97,-1.79,-1.63];
y = [2.88,2.83,1.36,0.897,0.703,0.428,0.525,0.873,2.052,2.88];
fill(x, y, [0.4902, 0.6863, 0.223], 'FaceAlpha', 0.05);
hold on; 
plot(x, y, 'Color', [0.4902, 0.6863, 0.223], 'LineWidth',1);
hold off;
% save(['H:\PHD\learning\research\AD_two_modal2\result\final_figure\normalize\kmeans\stability\k=3_stable.mat'], 'k3');
% load('E:\PHD\learning\research\AD_two_modal\result\multi_run\advanced_analysis\100roi\baseline\cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.6_pca_npi7_method_none_fmri4950_CAcomp4_fold10\label_diagnosis.mat');
% label2 = string(label);
% 
% figure('units','normalized','outerposition',[.2 .2 .25 .5]);
% box off;
% scatter(brain_feature(label2=='Dementia',1), brain_feature(label2=='Dementia',2), 'MarkerEdgeColor', 'white', 'MarkerFaceColor', [1, 0.702, 0.702]);
% hold on; 
% scatter(brain_feature(label2=="Cognitively normal",1), brain_feature(label2=="Cognitively normal",2), 'MarkerEdgeColor', 'white', 'MarkerFaceColor', [0.651, 0.8706, 0.9647]);
% legend('Dementia','Cognitively normal','Location','Best');
% hold off
