%% F9 for running selected lines
%% check for path  filename and extension

% moving image from screening microscope
moving_extension = ".png";
B_fname = 'p1_wA1_t1_m123_c0_z0_l1_o0';
M = imread("f:\experiment60-63_all_images\registration\mask\"+B_fname+moving_extension);

% Fixed image from Leica LMD6 63x
leica_extension = ".BMP";
L_fname = "p1_wA1_t1_m23_c1_z0_l1_o0_1";
F = imread("f:\datasets\211109-HK-60x\Extraannot\"+L_fname+leica_extension);


%% annot here
cpselect(M,F)

%% for loading annotation
% you can load your annotations to revise  with this command
% load("f:\datasets\211109-HK-60x\Extraannot\p1_wA1_t1_m23_c1_z0_l1_o0.mat");


%% Check annotations:
% show  the registered points
cpselect(M,F,movingPoints,fixedPoints)
%% transformation
% transform the image using the exported workspace min 2 points
% old transformation with rotaion translation and scaling
%tform = fitgeotrans(movingPoints,fixedPoints,'NonreflectiveSimilarity');
%% homography transformation sear and distrotion added
% min 4 points
% tform = fitgeotrans(movingPoints,fixedPoints,'projective')
% transform the image: if loaded from mat file (in this case regs.movingPoints, regs.fixedPoints)
% tform = fitgeotrans(regs.movingPoints,regs.fixedPoints,'NonreflectiveSimilarity')
%
Jregistered = imwarp(M,tform,'OutputView',imreFd(size(F)));
%% Here you can check the results remove the % from the line and run it with
%%f9
% show the transformed images with checkerboard
%figure, imshowpair(F,Jregistered, 'checkerboard')   
% other view types
% f = figure, imshowpair(F,Jregistered, 'blend')
f = figure, imshowpair(F,Jregistered, 'falsecolor')
 figure, imshowpair(F,Jregistered, 'montage')
%% this will save the points only with proper names
save(B_fname+".mat", 'fixedPoints','movingPoints')

%% here you can check the overlay and save as an image
fusedpair = imfuse(Jregistered, F);
imwrite(fusedpair, append(B_fname, '_combined.png'));
%% clean up after saving
clear fixedPoints  movingPoints




