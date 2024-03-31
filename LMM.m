clear
close all
file_name = 'measurement_lmm_daythre300_statethre6';
info = load(['H:\PHD\learning\research\AD_two_modal2\result\final_figure\normalize\kmeans\subtype\longitudinal\ad_onlyStable\', file_name, '.mat']);
% info = load(['H:\PHD\learning\research\AD_two_modal2\result\final_figure\normalize\kmeans\subtype\longitudinal\', file_name, '.mat']);
% info = load(['H:\PHD\learning\research\AD_two_modal\result\final_figure\normalize\kmeans\subtype\ad_only\longitudinal\', file_name, '.mat']);
Group_id = cellstr(info.subjects);
Group_id = string(Group_id)';
Label_raw = [info.subtypes{:}];
% Labels = cellstr(info.cdr_group);
% Labels = string(Labels);
Labels_dx = cellstr(info.baseline_dx_group);
Labels_dx = string(Labels_dx)';
% Labels = cellstr(info.baseline_dx_group);
% Labels = string(Labels)';
mask_dx = Labels_dx == 'Dementia';
Labels = [];
for i = 1:length(Label_raw)
    switch Label_raw(i)
        case 1
           Labels = [Labels; "Subtype 1"];
        case 2
           Labels = [Labels; "Subtype 2"];
        case 3
           Labels = [Labels; "Subtype 3"];
    end
end
% Labels(~mask_dx) = "HC";
% Labels(mask_dx) = "Dementia";
Week = [info.states{:}]';
save_name = strsplit(file_name, '_');
save_path = ['H:\PHD\learning\research\AD_two_modal2\result\final_figure\normalize\kmeans\subtype\longitudinal\ad_onlyStable\',...
    save_name{3}, '_', save_name{4}];
if isfolder(save_path)==0
     mkdir(save_path) 
end
items = ["NPIQINF"; "dx1"; "DELSEV"; "HALLSEV"; "AGITSEV"; "DEPDSEV"; ...
    "ANXSEV"; "ELATSEV"; "APASEV"; "DISNSEV"; "IRRSEV"; "MOTSEV"; "NITESEV"; "APPSEV";...
    "mmse"; "cdr"; "DIGIF"; "DIGIB"; 'ANIMALS'; 'VEG'; 'TRAILA'; 'TRAILALI';...
    'TRAILB'; 'TRAILBLI'; 'WAIS'; 'LOGIMEM'; 'MEMUNITS'; 'MEMTIME'; 'BOSTON'; "apoe";...
    'sumbox'; 'homehobb'; 'judgment'; 'memory'; 'orient'; 'perscare'; 'hyperactivity_subscale';...
    'anxiety_subscale'; 'npi_total'; 'SPEECH'; 'FACEXP'; 'TRESTFAC'; 'TRESTRHD'; 'TRESTLHD';...
    'TRESTRFT'; 'TRESTLFT'; 'TRACTRHD'; 'TRACTLHD'; 'RIGDNECK'; 'RIGDUPRT'; 'RIGDUPLF';...
    'RIGDLORT'; 'RIGDLOLF'; 'TAPSRT'; 'TAPSLF'; 'HANDMOVR'; 'HANDMOVL'; 'HANDALTR';...
    'HANDALTL'; 'LEGRT'; 'LEGLF'; 'ARISING'; 'POSTURE'; 'GAIT'; 'POSSTAB'; 'BRADYKIN';...
    'GDS'; 'BILLS'; 'TAXES'; 'SHOPPING'; 'GAMES'; 'STOVE'; 'MEALPREP'; 'EVENTS'; 'PAYATTN'; 'REMDATES'; 'TRAVEL'];    

for j = 3:length(items)
    if items(j) == 'GDS'
        a=0
    end
    depend_var = [];
    depend_var2 = [];
    if contains(items(j), 'SEV')
        for i = 1: length(Label_raw)
            depend_var = [depend_var; str2double(info.(items(j)){i})];
            depend_var2 = [depend_var2; str2double(info.NPIQINF{i})];% based on npiqinf to determine null or 0 value
        end
        mask = isnan(depend_var);
        mask2 = ~isnan(depend_var2);
        mask_ = mask & mask2;
        depend_var(mask_) = 0;
    elseif items(j) == 'hyperactivity_subscale' || items(j) == 'anxiety_subscale' || items(j) == 'npi_total'
            depend_var = [depend_var; info.(items(j))'];
    else
        for i = 1: length(Label_raw)
            depend_var = [depend_var; str2double(info.(items(j)){i})];
        end
    end
    
    if strcmp(items(j), 'apoe')
        a = 0;
        a
%         x = Week((Labels=="Subtype 1")&mask_dx);
%         y = depend_var((Labels=="Subtype 1")&mask_dx);
%         x = Week((Labels=="Subtype 1"));
%         y = depend_var((Labels=="Subtype 1"));
%         figure
%         plot(x,y);
    end

%     plot_pred_lmm(Group_id(mask_dx,:), depend_var(mask_dx,:), Labels(mask_dx,:), Week(mask_dx,:), items(j), save_path, file_name);
%     plot_pred_lmm_dx(Group_id(mask_dx,:), depend_var(mask_dx,:), Labels(mask_dx,:), Week(mask_dx,:), Group_id, depend_var, Labels_dx, Week, items(j), save_path, file_name);
    plot_pred_lmm(Group_id(~mask_dx,:), depend_var(~mask_dx,:), Labels(~mask_dx,:), Week(~mask_dx,:), items(j), save_path, file_name);
%     plot_pred_lmm(Group_id, depend_var, Labels, Week, items(j), save_path, file_name);
end
