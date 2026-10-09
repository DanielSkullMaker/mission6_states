clear;
N_final = 1;
N_K = 256;
N_L = 256;
% N_Long = N_K*N_L;
M_Vector = 200;
N_train_Recognition = size(1,2);
% X_xt = size(N_Long,M_Vector);
% A = size(N_Long,M_Vector);
% M_test_Kor_3 = zeros(N_Long,100,3);
N_end_train = 10;

    [M_test_3] = DATA_200(N_K, N_L, M_Vector);
    [M_train_3] = DATA_100(M_test_3, N_K, N_L, N_end_train);
    [M_train_3] = Otbor_100(N_K, N_L, M_test_Kor_3, N_end_train);
    [Sum_rate, Recogn_rate] = Recogn_VSPU_S1(M_test_3, M_train_3);
    
 
 N_train_Recognition(1) = N_end_train;
 N_train_Recognition(2) = Sum_rate;
 
 str = '„исло образцов и средн€€ точность';
 fprintf(str);
 disp(N_train_Recognition);
 disp(Recogn_rate);
 