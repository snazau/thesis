%%
load 'D:\Study\asp\thesis\implementation\data_orig\SVM_outliers_new_data\final_test_data.mat';
% thresh_arr=[0.1 0.05 0.025 0.01 0.005 0.0025 0.001 0.0005 0.0001 0.00005];
% thresh_arr=[0.01];
thresh_arr=[0.03];
num_th=0;
for thresh = thresh_arr
    num_th = num_th+1;
    clear acc precis spec sens;
    for subj=1:length(test)
        %subj=13;
        clear dat1;
        dat1 = test{1,subj};
        
        clear d_arr1 d_arr2 E y SVMModel score scoreGrid scorePred CVSVMModel status;
        clear st_arr;
        
        d_arr1 = dat1(:,1:31);
        % d_arr2 = dat1(:,32:281);
        %d_all = dat1(:,1:281);
        st_arr = test_mark{1,subj}(:);
        
        E(:,1) = mean(d_arr1,2);
        %E(:,2) = mean(d_arr2,2);
        
        y = ones(size(E,1),1);
        
        rng(1);
        SVMModel = fitcsvm(E,y,'KernelScale','auto','Standardize',true,...
            'OutlierFraction',thresh);
        
        % Plot figure
        % svInd = SVMModel.IsSupportVector;
        % h = 0.1; % Mesh grid step size
        % [X1,X2] = meshgrid(min(E(:,1)):h:max(E(:,1)),...
        %     min(E(:,2)):h:max(E(:,2)));
        % [~,score] = predict(SVMModel,[X1(:),X2(:)]);
        % scoreGrid = reshape(score,size(X1,1),size(X2,2));
        
        % figure
        % plot(E(:,1),E(:,2),'k.')
        % hold on
        % plot(E(svInd,1),E(svInd,2),'ro','MarkerSize',10)
        % contour(X1,X2,scoreGrid)
        % colorbar;
        % hold off
        
        CVSVMModel = crossval(SVMModel);
        [~,scorePred] = kfoldPredict(CVSVMModel);
        outlierRate(subj) = mean(scorePred<0);
        
        %%% Confusion matrix estimate
        clear tp_arr tp_arr_s tp ind fp_ind;
        
        ind = find(scorePred<0);
        
        %%% Save predicted positive indecies to *.mat file
        base_path = "D:\Study\asp\thesis\implementation\data_orig\SVM_outliers_new_data\results\";
        
        thresh_str = sprintf("%7.5f", thresh);
        thresh_dir_name = strrep(thresh_str, '.', '');
        thresh_dir = strcat(base_path, thresh_dir_name);
        mkdir(thresh_dir); 

        file_name_format = "%d.mat";
        file_name = sprintf(file_name_format, int32(subj));
        file_path = strcat(thresh_dir, '\', file_name);
        save(file_path, 'ind');
        %%%

        %%% Save targets to *.mat file
        targets_dir = strcat(base_path, 'targets');
        mkdir(targets_dir); 

        file_name_format = "%d.mat";
        file_name = sprintf(file_name_format, int32(subj));
        file_path = strcat(targets_dir, '\', file_name);
        save(file_path, 'st_arr');
        %%%

        clear seiz_ind;
        ns=0;
        for i=1:length(st_arr)
            if (st_arr(i)==1)
                ns=ns+1;
                seiz_ind(ns,1) = i;
            end;
        end;
        
        for i=1:ns
            %Taking into account neighbours
            tp_arr(:,i) = (seiz_ind(i)==ind)|(seiz_ind(i)-1==ind)|(seiz_ind(i)+1==ind)|(seiz_ind(i)-2==ind)|(seiz_ind(i)+2==ind);
            %tp_arr(:,i) = (seiz_ind(i)==ind);
        end;
        
        %FPs taking in account neighbours
        tp_arr_s = sum(tp_arr,2);
        tp_f=find(tp_arr_s>0);
        tp_arr_out{subj}=ind(tp_f);
        fp_ind = find(tp_arr_s==0);
        fp_arr{subj}=ind(fp_ind);
        fp=length(fp_arr{subj});
        for fi=1:(fp-1)
            if (fp_arr{subj}(fi)+1==fp_arr{subj}(fi+1))
                fp=fp-1;
            end;
        end;
        
        fp_per_h(subj)=fp/length(st_arr)*60;
        
        clear tp_arr_s;
        tp_arr_s = sum(tp_arr,1);
        tp=length(find(tp_arr_s>0));
        
        fn_ind = find(tp_arr_s==0);
        fn_arr{subj} = seiz_ind(fn_ind);
        fn=length(find(tp_arr_s==0));
        
        tn=length(find(scorePred>0))-fn;
        
        sens(subj)=tp/ns*100;
        spec(subj)=tn/(tn+fp)*100;
        precis(subj)=tp/(tp+fp)*100;
        acc(subj)=(tp+tn)/(tp+tn+fp+fn)*100;
        
        % Illustration
        
        % clear xp yp xs ys;
        % plot(E(:,1));
        % hold on; % hold the plot for other curves
        % xp = ind;
        % yp = E(xp,1);
        % plot(xp,yp,'o','MarkerSize',10);
        % hold on;
        % xs = seiz_ind;
        % ys = E(xs,1);
        % plot(xs,ys,'go','MarkerSize',5);
        % hold off;
        % % % 
        % clear xp yp xs ys;
        % figure;
        % plot(E(:,2));
        % hold on;
        % xp = ind;
        % yp = E(xp,2);
        % plot(xp,yp,'o','MarkerSize',10);
        % hold on;
        % xs = seiz_ind;
        % ys = E(xs,2);
        % plot(xs,ys,'go','MarkerSize',5);
        % hold off;
        
    end;
    acc_m=mean(acc);
    sens_m=mean(sens);
    spec_m=mean(spec);
    precis_m=mean(precis);
    fp_per_h_m=mean(fp_per_h);
    outl_m = mean(outlierRate);

    file_path = strcat(thresh_dir, '\', file_name);
    save(strcat(thresh_dir, '\', 'accuracy_1_by_1_2En_ranges_fr=0.01.txt'), 'acc', '-ascii');
    save(strcat(thresh_dir, '\', 'sensitivity_1_by_1_2En_ranges_fr=0.01.txt'), 'sens', '-ascii');
    save(strcat(thresh_dir, '\', 'specificity_1_by_1_2En_ranges_fr=0.01.txt'), 'spec', '-ascii');
    save(strcat(thresh_dir, '\', 'precision_1_by_1_2En_ranges_fr=0.01.txt'), 'precis', '-ascii');
    save(strcat(thresh_dir, '\', 'precision_1_by_1_2En_ranges_fr=0.01.txt'), 'precis', '-ascii');

    acc_m_str = sprintf("acc_m = %6.4f", acc_m);
    sens_m_str = sprintf("sens_m = %6.4f", sens_m);
    spec_m_str = sprintf("spec_m = %6.4f", spec_m);
    precis_m_str = sprintf("precis_m = %6.4f", precis_m);
    fp_per_h_m_str = sprintf("fp_per_h_m = %6.4f", fp_per_h_m);
    outl_m_str = sprintf("outl_m = %6.4f", outl_m);
    save(strcat(thresh_dir, '\', 'acc_m.txt'), 'acc_m', '-ascii')
    save(strcat(thresh_dir, '\', 'sens_m.txt'), 'sens_m', '-ascii')
    save(strcat(thresh_dir, '\', 'spec_m.txt'), 'spec_m', '-ascii')
    save(strcat(thresh_dir, '\', 'precis_m.txt'), 'precis_m', '-ascii')
    save(strcat(thresh_dir, '\', 'fp_per_h_m.txt'), 'fp_per_h_m', '-ascii')
    save(strcat(thresh_dir, '\', 'outl_m.txt'), 'outl_m', '-ascii')

    out_arr{num_th}.thresh = thresh;
    out_arr{num_th}.acc_arr = acc;
    out_arr{num_th}.acc_m = acc_m;
    out_arr{num_th}.sens_arr = sens;
    out_arr{num_th}.sens_m = sens_m;
    out_arr{num_th}.spec_arr = spec;
    out_arr{num_th}.spec_m = spec_m;
    out_arr{num_th}.precis_arr = precis;
    out_arr{num_th}.precis_m = precis_m;
    out_arr{num_th}.fp_per_h_arr = fp_per_h;
    out_arr{num_th}.fp_per_h_m = fp_per_h_m;
    out_arr{num_th}.outlierRate_arr = outlierRate;
    out_arr{num_th}.outlierRate_m = outl_m;
    out_arr{num_th}.tp_arr = tp_arr_out;
    out_arr{num_th}.fp_arr = fp_arr;
    out_arr{num_th}.fn_arr = fn_arr;
    
end;
%%
% save('D:\Study\asp\thesis\implementation\data_orig\SVM_outliers_new_data\results\accuracy_1_by_1_2En_ranges_fr=0.01.txt', 'acc', '-ascii');
% save('D:\Study\asp\thesis\implementation\data_orig\SVM_outliers_new_data\results\sensitivity_1_by_1_2En_ranges_fr=0.01.txt', 'sens', '-ascii');
% save('D:\Study\asp\thesis\implementation\data_orig\SVM_outliers_new_data\results\specificity_1_by_1_2En_ranges_fr=0.01.txt', 'spec', '-ascii');
% save('D:\Study\asp\thesis\implementation\data_orig\SVM_outliers_new_data\results\precision_1_by_1_2En_ranges_fr=0.01.txt', 'precis', '-ascii');

%%
% plot([0.01:0.01:0.5],spec_m);
% hold on;
% plot([0.01:0.01:0.5],precis_m);

%%
% for i=1:10
%     pl(i) = out_arr{i}.sens_m;
% end;
% figure;
% semilogx(thresh_arr*100,pl);