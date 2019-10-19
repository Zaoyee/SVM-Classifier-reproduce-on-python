%%
% * svm ??????
%
%% ????
% * ??data???m*n?m????n??
% * label:m*1  ?????-1?1???
clc
clear
T = readtable('ESLmixture.csv');
data = T{:,[2,3,4]};

train_data = data(:,[1,2]);
label = data(:,end);
label(find(label==0)) = -1;
[num_data,d] = size(train_data);
data = train_data;
%%
alphas = zeros(num_data,1);
b = 0;
%error = zeros(num_data,2);
tol = 0.0001;
C = 0.6;
iter = 0;
max_iter = 40;

alpha_change = 0;
entireSet = 1;%???????????????????
while (iter < max_iter) && ((alpha_change > 0) || entireSet)
    alpha_change = 0;
    %% -----------?????-------------------------
    if entireSet 
        for i = 1:num_data
            Ei = calEk(data,alphas,label,b,i);%????
            if (label(i)*Ei<-0.001 && alphas(i)<C)||...
                    (label(i)*Ei>0.001 && alphas(i)>0)
                %?????alphas
                [j,Ej] = select(i,data,num_data,alphas,label,b,C,Ei,entireSet);
                alpha_I_old = alphas(i);
                alpha_J_old = alphas(j);
                if label(i) ~= label(j)
                    L = max(0,alphas(j) - alphas(i));
                    H = min(C,C + alphas(j) - alphas(i));
                else
                    L = max(0,alphas(j) + alphas(i) -C);
                    H = min(C,alphas(j) + alphas(i));
                end
                if L==H
                    continue;end
                eta = 2*data(i,:)*data(j,:)'- data(i,:)*...
                    data(i,:)' - data(j,:)*data(j,:)';
                if eta >= 0 
                    continue;end
                alphas(j) = alphas(j) - label(j)*(Ei-Ej)/eta;
                %????
                if alphas(j) > H
                    alphas(j) = H;
                elseif alphas(j) < L
                    alphas(j) = L;
                end
                if abs(alphas(j) - alpha_J_old) < 1e-4
                    continue;end
                alphas(i) = alphas(i) + label(i)*...
                    label(j)*(alpha_J_old-alphas(j));
                b1 = b - Ei - label(i)*(alphas(i)-alpha_I_old)*...
                    data(i,:)*data(i,:)'- label(j)*...
                    (alphas(j)-alpha_J_old)*data(i,:)*data(j,:)';
                b2 = b - Ej - label(i)*(alphas(i)-alpha_I_old)*...
                    data(i,:)*data(j,:)'- label(j)*...
                    (alphas(j)-alpha_J_old)*data(j,:)*data(j,:)';
                if (alphas(i) > 0) && (alphas(i) < C)
                    b = b1;
                elseif (alphas(j) > 0) && (alphas(j) < C)
                    b = b2;
                else
                    b = (b1+b2)/2;
                end
                alpha_change = alpha_change + 1;
            end
        end
         iter = iter + 1;
   %% --------------????(alphas=0~C)???--------------------------
    else
        index = find(alphas>0 & alphas < C);
        for ii = 1:length(index)
            i = index(ii);
            Ei = calEk(data,alphas,label,b,i);%????
            if (label(i)*Ei<-0.001 && alphas(i)<C)||...
                    (label(i)*Ei>0.001 && alphas(i)>0)
                %???????
                [j,Ej] = select(i,data,num_data,alphas,label,b,C,Ei,entireSet);
                alpha_I_old = alphas(i);
                alpha_J_old = alphas(j);
                if label(i) ~= label(j)
                    L = max(0,alphas(j) - alphas(i));
                    H = min(C,C + alphas(j) - alphas(i));
                else
                    L = max(0,alphas(j) + alphas(i) -C);
                    H = min(C,alphas(j) + alphas(i));
                end
                if L==H
                    continue;end
                eta = 2*data(i,:)*data(j,:)'- data(i,:)*...
                    data(i,:)' - data(j,:)*data(j,:)';
                if eta >= 0
                    continue;end
                alphas(j) = alphas(j) - label(j)*(Ei-Ej)/eta;  
                %????
                if alphas(j) > H
                    alphas(j) = H;
                elseif alphas(j) < L
                    alphas(j) = L;
                end
                if abs(alphas(j) - alpha_J_old) < 1e-4
                    continue;end
                alphas(i) = alphas(i) + label(i)*...
                    label(j)*(alpha_J_old-alphas(j));
                b1 = b - Ei - label(i)*(alphas(i)-alpha_I_old)*...
                    data(i,:)*data(i,:)'- label(j)*...
                    (alphas(j)-alpha_J_old)*data(i,:)*data(j,:)';
                b2 = b - Ej - label(i)*(alphas(i)-alpha_I_old)*...
                    data(i,:)*data(j,:)'- label(j)*...
                    (alphas(j)-alpha_J_old)*data(j,:)*data(j,:)';
                if (alphas(i) > 0) && (alphas(i) < C)
                    b = b1;
                elseif (alphas(j) > 0) && (alphas(j) < C)
                    b = b2;
                else
                    b = (b1+b2)/2;
                end
                alpha_change = alpha_change + 1;
            end
        end
        iter = iter + 1;
    end
    %% --------------------------------
    if entireSet %??????????????????
        entireSet = 0;
    elseif alpha_change == 0 
        %??????????????????alpha???????
        entireSet = 1;
    end
    disp(['iter ================== ',num2str(iter)]);    
end
%% ????W
W = (alphas.*label)'*data;
%????????
index_sup = find(alphas ~= 0);
%??????
predict = (alphas.*label)'*(data*data') + b;
predict = sign(predict);
%% ????
figure;
index1 = find(predict==-1);
data1 = (data(index1,:))';
plot(data1(1,:),data1(2,:),'+r');
hold on
index2 = find(predict==1);
data2 = (data(index2,:))';
plot(data2(1,:),data2(2,:),'*');
hold on
dataw = (data(index_sup,:))';
plot(dataw(1,:),dataw(2,:),'og','LineWidth',2);
% ????????b????1????
hold on
k = -W(1)/W(2);
x = 1:0.1:5;
y = k*x + b;
plot(x,y,x,y-1,'r--',x,y+1,'r--');
title(['??????C = ',num2str(C)]);





function Ek = calEk(data,alphas,label,b,k)
pre_Li = (alphas.*label)'*(data*data(k,:)') + b;
Ek = pre_Li - label(k);
end


function [J,Ej] = select(i,data,num_data,alphas,label,b,C,Ei,choose)
maxDeltaE = 0;maxJ = -1;
if choose == 1 %???---????alphas
    j = randi(num_data ,1);
    if j == i
        temp = 1;
        while temp
            j = randi(num_data,1);
            if j ~= i
                temp = 0;
            end
        end
    end
    J = j;
    Ej = calEk(data,alphas,label,b,J);
else %????--??????alphas
    index = find(alphas>0 & alphas < C);
    for k = 1:length(index)
        if i == index(k)
            continue;
        end
        temp_e = calEk(data,alphas,label,b,k);
        deltaE = abs(Ei - temp_e); %???Ei?????alphas
        if deltaE > maxDeltaE
            maxJ = k;
            maxDeltaE = deltaE;
            Ej = temp_e;
        end
    end
    J = maxJ;
end
end