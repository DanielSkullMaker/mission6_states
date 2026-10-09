function [M_train_3] = Otbor_100(N_K, N_L, M_test_Kor_3, N_end_train)
N_Vektor = N_K*N_L;
% M_Vector_200 = 200;
M_Vector_100 = 100;
M_train_3 = zeros(N_Vektor,M_Vector_100,3);
% Y_XX = size(M_Vector_100, M_Vector_100);
% M_test_100_3 = size(N_Vektor,M_Vector_100,3);
% X_xt = size(N_Vektor,M_Vector_100);
% A_train = size(N_Vektor,M_Vector_100);
R_Start = 1;

for k_T = 1:1:3
   X_xt = M_test_Kor_3(:,:,k_T);
   A_train = X_xt;
% for Nklass = 1:1:N_end_train
%  for M_klass = 1:1:N_end_train  
Nklass = 1;
if Nklass <= N_end_train;
  for k_test = Nklass:1:M_Vector_100
       X_tmp = A_train(:,k_test);
     if k_test == 1
       M_train = X_tmp;
       M_train_tmp = M_train;
     end
     if k_test ~= 1
        Y_R = M_train'*X_tmp;
        Y_L = Y_R';
        Y_XX = inv(M_train'*M_train);
        R = (Y_L*Y_XX*Y_R)/(X_tmp'*X_tmp);

      if R_Start > R
            R_Start = R;
%             k_min = k_test;
            V_min = X_tmp; 
      end 
     end
     Nklass = Nklass + 1;
  end
        M_train = [M_train_tmp, V_min];
        M_train_tmp = M_train;
%        Nklass = Nklass + 1;
 end
    M_train_3(:,:,k_T) = M_train;   
end
end
 