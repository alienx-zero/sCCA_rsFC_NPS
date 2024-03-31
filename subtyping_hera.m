clear
close all

% load('E:\PHD\learning\research\AD_two_modal\result\advanced_analysis\200roi\baseline\cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.5_fa_npi7_method_pca_fmri90_CAcomp3_fold10\brain_transformed.mat')
% load('E:\PHD\learning\research\AD_two_modal\result\advanced_analysis\200roi\baseline\cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.5_fa_npi7_method_pca_fmri90_CAcomp3_fold10\npi_transformed.mat')
load('E:\PHD\learning\research\AD_two_modal\result\multi_run\advanced_analysis\100roi\baseline\cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.6_pca_npi7_method_none_fmri4950_CAcomp4_fold10\brain_transformed.mat')
load('E:\PHD\learning\research\AD_two_modal\result\multi_run\advanced_analysis\100roi\baseline\cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.6_pca_npi7_method_none_fmri4950_CAcomp4_fold10\npi_transformed.mat')


% clustFcn = @(X,K) kmeans(X, K,'Replicates',5);
% linkage(brain_feature(:,1:2), 'ward');
clustFcn = @(X,K) cluster(linkage(X, 'ward', 'cosine'),'Maxclust',K);
% clustFcn = @(X,K) linkage(X,'Maxclust',K);

gap = [];
DaviesBouldin = [];
Silhouette = [];
CalinskiHarabasz = [];
labels = zeros(length(brain_feature), 100);
for i = 1:10
    i
    idx = 1:length(brain_feature);
    idx2 = randperm(length(brain_feature));
    mask = idx2(1:round(length(brain_feature)*0.9));
    evaluation1 = evalclusters(brain_feature(mask,1:2),clustFcn,"gap","KList",2:10);
    evaluation2 = evalclusters(brain_feature(mask,1:2),clustFcn,"DaviesBouldin","KList",2:10);
    evaluation3 = evalclusters(brain_feature(mask,1:2),clustFcn,"Silhouette","KList",2:10);
    evaluation4 = evalclusters(brain_feature(mask,1:2),clustFcn,"CalinskiHarabasz","KList",2:10);
    gap = [gap; evaluation1.CriterionValues];
    DaviesBouldin = [DaviesBouldin; evaluation2.CriterionValues];
    Silhouette = [Silhouette; evaluation3.CriterionValues];
    CalinskiHarabasz = [CalinskiHarabasz; evaluation4.CriterionValues];
%     Z = linkage(brain_feature(mask,1:2),'ward', 'cosine');
%     c = cluster(Z,'Maxclust',3);
%     labels(mask, i) = c;
end

gap_mean = mean(gap);
gap_std = std(gap);
DaviesBouldin_mean = mean(DaviesBouldin);
DaviesBouldin_std = std(DaviesBouldin);
Silhouette_mean = mean(Silhouette);
Silhouette_std = std(Silhouette);
CalinskiHarabasz_mean = mean(CalinskiHarabasz);
CalinskiHarabasz_std = std(CalinskiHarabasz);

error_plot(2:10, gap_mean, gap_std, 'Gap value', 'Number of clusters', 0.2, 1, 1.5, 10.5);
error_plot(2:10, DaviesBouldin_mean, DaviesBouldin_std, 'Davies–Bouldin index',  'Number of clusters',0.9, 1.9, 1.5, 10.5);
error_plot(2:10, Silhouette_mean, Silhouette_std, 'Silhouette coefficient', 'Number of clusters', 0.1, 0.5, 1.5, 10.5);
error_plot(2:10, CalinskiHarabasz_mean, CalinskiHarabasz_std, 'Calinski-Harabasz criterion', 'Number of clusters', 40, 90, 1.5, 10.5);

% corr(brain_feature, npi_feature);
figure(1)
Z = linkage(brain_feature(:,1:2),'ward', 'cosine');
c = cluster(Z,'Maxclust',2);
dendrogram(Z, 0, 'ColorThreshold','default')
% 
figure(2)
Z2 = linkage(npi_feature(:,1:2),'ward', 'cosine');
c2 = cluster(Z,'Maxclust',2);
dendrogram(Z2, 0, 'ColorThreshold','default')

cgo = clustergram(brain_feature(:,1:2), 'Colormap', colormap(othercolor('Spectral11')), 'Linkage', 'ward', 'Cluster', 'column',...
    'ColumnPDist', 'cosine', 'RowPDist', 'cosine','LabelsWithMarkers', 'true', 'DisplayRatio', [0.2,0.8],...
    'OptimalLeafOrder', 'true');
cgroup1 = clusterGroup(cgo,1,'row');
cgroup2 = clusterGroup(cgo,2,'row');
cgroup3 = clusterGroup(cgo,3,'row');
cgroup4 = clusterGroup(cgo,4,'row');
row_id = cgo.RowLabels;  
subj_idx = {};
subtype = {};
color = {};
for i = 1:5
    subtype{i} = ['Subtype', int2str(c(i))]; 
    subj_idx{i} = str2num(row_id{i}); 
    if i <= 40
        color{i} = 'r'; 
    elseif i > 40 && i < 80 
        color{i} = 'b'; 
    else
        color{i} = 'g'; 
    end
end
rm = struct('GroupNumber',{10,50},'Annotation',{'a', 'b'},...
     'Color',{'r','b'});
set(cgo,'RowGroupMarker',rm)
% set(cgo,'Dendrogram',3, 'DisplayRatio', [0.2,0.8])
certer_mean = zeros(100, 3, 3);
for i =1:100
    for j = 1:3
        certer_mean(i,j,:) = mean(brain_feature(labels(:,i)==j, :));
    end
end
certer_mean1 = squeeze(certer_mean(:,1,1:2));
certer_mean2 = squeeze(certer_mean(:,2,1:2));
certer_mean3 = squeeze(certer_mean(:,3,1:2));
%%
% resort center
labels_resort = zeros(length(brain_feature), 100);

for i =1:100
    label1_raw = squeeze(labels(:,1));
    label2_raw = squeeze(labels(:,i));
    mask = label1_raw~=0 & label2_raw ~= 0;
    label1 = label1_raw(mask);
    label2 = label2_raw(mask);
    if i == 1
        labels_resort(:,i) = label1_raw;
        continue
    end

    similarity = pdist2(label1',label2','jaccard');
    labels_resort(:,i) = label2_raw;
    for j = 1:3
        for k = 1:3
            if j~=k
                label3_raw = squeeze(labels(:,i));
                label3 = label3_raw(mask);
                label3(label2==j)=k;
                label3(label2==k)=j;
                label3_raw(label2_raw==j)=k;
                label3_raw(label2_raw==k)=j;
                similarity1 = pdist2(label1',label3','jaccard');
                if similarity1 < similarity
                    labels_resort(:,i) = label3_raw;
                    similarity = similarity1;
                end
                label3_raw = squeeze(labels(:,i));
                label3 = label3_raw(mask);
                label3(label2==j)=k;
                C = setdiff([1,2,3],[k,j]);
                label3(label2==k)=C;
                label3(label2==C)=j;
                label3_raw(label2_raw==j)=k;
                label3_raw(label2_raw==k)=C;
                label3_raw(label2_raw==C)=j;

                similarity2 = pdist2(label1',label3','jaccard');
                if similarity2 < similarity
                    labels_resort(:,i) = label3_raw;
                    similarity = similarity2;
                end
            end
        end
    end
end
certer_mean_resort = zeros(100, 3, 2);

for i =1:100
    for j = 1:3
        certer_mean_resort(i,j,:) = mean(brain_feature(labels_resort(:,i)==j, 1:2));
    end
end

certer_mean_resort1 = squeeze(certer_mean_resort(:,1,:));
certer_mean_resort2 = squeeze(certer_mean_resort(:,2,:));
certer_mean_resort3 = squeeze(certer_mean_resort(:,3,:));

figure('units','normalized','outerposition',[.2 .2 .25 .5]);
box off;
scatter(certer_mean_resort1(:,1), certer_mean_resort1(:,2), 'Color', [1, 0.702, 0.702]);
hold on; 
scatter(certer_mean_resort2(:,1), certer_mean_resort2(:,2), 'Color', [0.651, 0.8706, 0.9647]);
hold on; 
scatter(certer_mean_resort3(:,1), certer_mean_resort3(:,2), 'Color', [.4902, 0.6863, 0.223]);
hold off; 

save('E:\PHD\learning\research\AD_two_modal\result\advanced_analysis\200roi\baseline\cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.5_fa_npi7_method_pca_fmri90_CAcomp3_fold10\cluster\label.mat', 'c')
%%
% resort center
% for i =1:100
%     center1 = squeeze(certer_mean(i,:,1:2));
%     center2 = squeeze(certer_mean(1,:,1:2));
%     similarity = pdist2(center1,center2);
%     [val,ind] = sort(similarity);
%     ind_sort = ind(1,:);
%     for j =1:3
%         labels_resort(labels(:,i)==j,i) = ind_sort(j);
%         certer_mean_resort(i,j,:) = certer_mean(i,ind_sort(j),1:2);
%     end
%     certer_mean_resort1 = squeeze(certer_mean_resort(:,1,:));
%     certer_mean_resort2 = squeeze(certer_mean_resort(:,2,:));
%     certer_mean_resort3 = squeeze(certer_mean_resort(:,3,:));
% end


% % similarity = jaccard(c,c2);
% similarity2 = pdist2(c',c2','jaccard');
% 
