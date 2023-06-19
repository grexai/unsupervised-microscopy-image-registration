
%% HK 60-63x dataset

base_path = 'd:\datasets\Image_registration\211109-HK-60x\splitted\'


sets = ["test","train"]
for set=1:2
    outfolder = append(base_path,'train\annot_scaled\')
    mkdir(outfolder)
    mkdir(append(outfolder, 'overlay'))
    mkdir(append(outfolder, 'masks'))
    mkdir(append(outfolder, 'warped_bias'))
    mkdir(append(outfolder, 'warped_mask'))

    folder  = append(base_path, sets(set),'\annotation\')
    list = dir( append(folder, '*.mat'));

    range = numel(list)
    for idx=1:range

        %reg = load("d:\datasets\Image_registration\211109-HK-60x\splitted\train\annotation\"+list(idx).name);
        reg = load(append(folder,list(idx).name));
        re = regexp(list(idx).name,'\d*','Match');
        %p1_wA1_t1_m1_c1_z0_l1_o0.png
        M_fname = append('p1_wA1_t1_m',re{1,4},'_c1_z0_l1_o0'); % c1
        M = imread(append(base_path ,sets(set) ,"\registration\", M_fname, ".png"));
        %% MASK
        M_fname = append('p1_wA1_t1_m',re{1,4},'_c0_z0_l1_o0');
        M2 = imread(append("d:\datasets\Image_registration\211109-HK-60x\registration\mask\", M_fname, ".png"));
%         M2 = uint8(M2);
%         M2 = edge(M2);
%         kernel = strel('square',5);
%         M = imdilate(M, kernel);
%         % Leica Fixed image
        %% leica
        F_fname = append('p1_wA1_t1_m',re{1,4},'_c1_z0_l1_o0');
        F = imread(append(base_path,sets(set),"\lmd63x\",F_fname, "_1.BMP"));
        %
        F = F(1:1440,1:1440,:);
        M = M(1:850,1:850,:);
        M2 = M2(1:850,1:850,:);
         %M = imresize(M,[850 , 850]);
        %% unscaled tf needed for alignannotated
        fixedPoints = reg.fixedPoints;
        movingPoints = reg.movingPoints;% now we do that here
        tform = fitgeotrans(movingPoints,fixedPoints,'nonreflectivesimilarity');
        % Tm = tform.T;
        Jregistered = imwarp(M,tform,'OutputView',imref2d(size(F)));
        imwrite(Jregistered, append(outfolder,'warped_bias\',F_fname,'_ws.png'));
        
         
        %%  scaling needed for visualization and eval
        F = imresize(F,[1024 , 1024]);
        M = imresize(M,[1024 , 1024]);
        M2 = imresize(M2,[1024 , 1024]);
        
        imwrite(M2, append(outfolder,'\masks\',list(idx).name(1:end-4),'.png'));
        %% the train images converted to 1024 scale down
        fixedPoints = reg.fixedPoints*0.7111;
        movingPoints = reg.movingPoints.*1.2;% now we do that here
        tform = fitgeotrans(movingPoints,fixedPoints,'nonreflectivesimilarity');
        Tm = tform.T;
        % display(num2str(idx))
        %%
        Jregistered = imwarp(M,tform,'OutputView',imref2d(size(F)));
        Jregistered2 = imwarp(M2,tform,'OutputView',imref2d(size(F)));
        fusedpair = imfuse(Jregistered, F);
        imwrite(Jregistered, append(outfolder,'warped_bias\',F_fname,'_w.png'));
        imwrite(Jregistered2, append(outfolder,'warped_mask\',F_fname,'_w.png'));
        
        %asd = Jregistered(:,:,1)  
        ol = imoverlay(F,Jregistered(:,:,1));
        theta_recovered = atan2(Tm(2,1),Tm(1,1))*180/pi;
        imwrite(ol, append(outfolder,'overlay\',list(idx).name(1:end-4),'_OL.png'));
        save(append(outfolder, (list(idx).name(1:end-4)+".mat")), 'Tm');
        display(num2str(idx/range*100))
        %imwrite(Jregistered, append(outfolder,'\',list(idx).name(1:end-4),'_w_s.png'));

    end 
end

