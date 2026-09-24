function [M_train_3] = DATA_100(M_test_3, N_K, N_L, N_end_train)

N_Vektor = N_K*N_L;
M_Vector_200 = 200;
M_Vector_100 = 100;
M_train_3 = zeros(N_Vektor,N_end_train,3);
R_Start = 1;

% X_xt = size(N_Vektor,M_Vector_200);
% A_tmp = size(N_Vektor,M_Vector_200);
% Prizn_Start = 0;
% R_initial=0;

for k_T = 1:1:3
   X_xt = M_test_3(:,:,k_T);
   A_tmp = X_xt;
   Prizn_Start = 0;
%    k_Start_100 = size(A_tmp,2);
%    N_data = M_Vector_200;
%    k_end_100 = M_Vector_100 + 1;
%    R_initial = 0;
% while k_Start_100 > M_Vector_100
% if k_Start_100 > M_Vector_100

for   N_data = 1:1:M_Vector_100
% if N_data >= M_Vector_100
   R_initial=0;
   j_end = size(A_tmp,2);
   i_end = size(A_tmp,2) - 1;
for i = 1:1:i_end
     for j = i+1:1:j_end
            R_edu = corrcoef(A_tmp(:,i),A_tmp(:,j));
            R = R_edu(1,2);
            if R > R_initial
               R_initial = R;
               i_out = i;
               j_out = j;
            end
      end         
 end
        X_tmp = A_tmp(:,j_out);
        A_tmp(:,j_out) = [];
%         N_data = size(A_tmp,2);
%         N_data = N_data - 1;
%         k_Start_100 = size(A_tmp,2);
%         N_Min = size(A,2);
        if Prizn_Start > 0.5
            Y = [Y_temp,X_tmp];
            Y_temp = Y;
        end
        if Prizn_Start == 0
             Y_temp = X_tmp;
             Prizn_Start = 1;
        end
        
     end
     
%   Отбор удаленных друг от друга ______________________________   
       A_train = Y;
       Nklass_end =  N_end_train - 1;
for Nklass = 1:1:Nklass_end;
     R_Start = 1;
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
%          k_min = k_test;
         V_min = X_tmp; 
      end   
     end    
    end      
        M_train = [M_train_tmp, V_min];
        M_train_tmp = M_train; 
   end
        M_train_3(:,:,k_T) = M_train;        
   end  
 end
  
