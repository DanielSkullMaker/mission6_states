function [M_test_3] = DATA_200(N_K, N_L, M_Vector)
% clear;
n = 1;
N_Vektor = N_K*N_L;
% M_Vector = 200;
V_train = [];

% Sum = 0;
% Sum_Recogn = 0;
% Recogn  = 0;
% NoRecogn = 0;
% M_train_3 = zeros(N_Vektor,M_Vector,3);
M_test_3 = zeros(N_Vektor,M_Vector,3);

% Перебор тестовых _________________________________________________

% Класс 1 (Test_1) _________________________________________________
k_T = 1;
        imgPath_1_test = 'TEST0/Test_1/';
        images  = dir([imgPath_1_test  '*.png']);
        N_test = length(images);
    for k_test = 1:1:N_test
        Image_tmp = imread([imgPath_1_test images(k_test).name]);
        X_tmp_test = double(Image_tmp)/255;       

% Класс 1 (Test_1)_________________________________________________  
N_Vect_tst = N_K*N_L;
V_test = zeros(N_Vect_tst,1);
n=1;
for i = 1:1:N_K
    for j = 1:1:N_L
        V_test(n,1) =  X_tmp_test(i,j);
        n = n + 1;   
    end
end

if k_test == 1
M_test1 = V_test;
end
if k_test ~= 1
M_tmp_test1 = [M_test1,V_test];
M_test1 = M_tmp_test1;   
end
end
M_test_3(:,:,k_T) = M_test1; 

% Класс 2 (Test_2) _________________________________________________
k_T = 2;
        imgPath_2_test = 'TEST0/Test_2/';
        images  = dir([imgPath_2_test  '*.png']);
        N_test = length(images);
    for k_test = 1:1:N_test
        Image_tmp = imread([imgPath_2_test images(k_test).name]);
        X_tmp_test = double(Image_tmp)/255;       

n=1;
for i = 1:1:N_K
    for j = 1:1:N_L
        V_test(n,1) =  X_tmp_test(i,j);
        n = n + 1;   
    end
end

if k_test == 1
M_test2 = V_test;
end
if k_test ~= 1
M_tmp_test2 = [M_test2,V_test];
M_test2 = M_tmp_test2;   
end
end
M_test_3(:,:,k_T) = M_test2; 

% Класс 3 (Test_3) _________________________________________________
k_T = 3;
        imgPath_3_test = 'TEST0/Test_3/';
        images  = dir([imgPath_3_test  '*.png']);
        N_test = length(images);
    for k_test = 1:1:N_test
        Image_tmp = imread([imgPath_3_test images(k_test).name]);
        X_tmp_test = double(Image_tmp)/255;       
n=1;
for i = 1:1:N_K
    for j = 1:1:N_L
        V_test(n,1) =  X_tmp_test(i,j);
        n = n + 1;   
    end
end

if k_test == 1
M_test3 = V_test;
end
if k_test ~= 1
M_tmp_test3 = [M_test3,V_test];
M_test3 = M_tmp_test3;   
end
end
M_test_3(:,:,k_T) = M_test3; 
end
