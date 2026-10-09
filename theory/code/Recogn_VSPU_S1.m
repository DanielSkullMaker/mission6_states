function [Sum_rate, Recogn_rate] = Recogn_VSPU_S1(M_test_3, M_train_3)
% clear;
Recogn = zeros(1,3);
Sum_class = zeros(1,3);
NoRecogn = zeros(1,3);
% Recogn_rate = zeros(1,3);
% NoRecogn_rate = zeros(1,3);

Recogn_rate = [];
NoRecogn_rate = [];

Sum = 0;
Sum_Recogn = 0;

% Начало ______________________________________
for k=1:1:3 
     Recogn(k)=0;
     NoRecogn(k)=0;
      X_xt = M_test_3(:,:,k); 
 % _____________________________________________________
        for i=1:1:size(X_xt,2)
            xt = X_xt(:,i);
%Усреднение 3_____________________________________________
            if i == 2 || i == (size(X_xt,2) - 1)
                xt_1 = X_xt(:,i-1);
                xt_2 = X_xt(:,i+1);
                xt_temp = (xt_1 + xt + xt_2)/3;
                xt = xt_temp;
            end             
%Усреднение 5_____________________________________________
            if i == 3 || i == (size(X_xt,2) - 2)                    
                xt_0 = X_xt(:,i-2);
                xt_1 = X_xt(:,i-1);
                xt_2 = X_xt(:,i+1);
                xt_3 = X_xt(:,i+2);
                xt_temp = (xt_0 + xt_1 + xt + xt_2 + xt_3)/5;
                xt = xt_temp;
            end
 %________________________________________________________________ 
            if i >= 4 && i <= (size(X_xt,2) - 3)                      
                xt_0 = X_xt(:,i-3);
                xt_1 = X_xt(:,i-2);
                xt_2 = X_xt(:,i-1);
                xt_3 = X_xt(:,i+1);
                xt_4 = X_xt(:,i+2);
                xt_5 = X_xt(:,i+3);
                xt_temp = (xt_0 + xt_1 + xt_2 + xt + xt_3 + xt_4 + xt_5)/7;
                xt = xt_temp;
            end
 %________________________________________________________________
 
   R_sum = zeros(1,3);
   for l=1:1:3  
      X = M_train_3(:,:,l); 
%       R_Start = 0;
%       X_1 = X;
            Y_R=X'*xt;
            Y_L=Y_R';
            Y_XX=inv(X'*X);
            R =(Y_L*Y_XX*Y_R)/(xt'*xt);       
 
                    R_sum(l) = R;
        end
                   
                [R_conj,Index] = sort(R_sum);
                label = Index(3);
                if label == k
                Recogn(k) = Recogn(k)+1;
                else
                NoRecogn(k) = NoRecogn(k)+1;
                end
                
        end
            
            Sum_Recogn = Sum_Recogn + Recogn(k);
            Sum_class(k) = Recogn(k) + NoRecogn(k);
            Sum = Sum + Sum_class(k);
            Recogn_rate(k)= Recogn(k)/Sum_class(k);
            NoRecogn_rate(k)= NoRecogn(k)/Sum_class(k);
            
end
        Sum_rate = Sum_Recogn/Sum; 
end
