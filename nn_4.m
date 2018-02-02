% This function is the primary driver for homework 3 part 1
function l3a
close all;
clear all;
clc;
% centroid for the three classes
c1=[1 1];
c2=[3 1];
c3=[2 3];

sd=0.2;

% number of data points per class
N=100;

rand('seed', 1);

% generate data points for the three classes
x1=randn(N,2)*sd+ones(N,1)*c1;
x2=randn(N,2)*sd+ones(N,1)*c2;
x3=randn(N,2)*sd+ones(N,1)*c3;

% generate the labels for the three classes in the binary notation
y1= repmat([1 0 0],N,1);
y2= repmat([0 1 0],N,1);
y3= repmat([0 0 1],N,1);

% creating the test data points
a1min = min([x1(:,1);x2(:,1);x3(:,1)]);
a1max = max([x1(:,1);x2(:,1);x3(:,1)]);

a2min = min([x1(:,2);x2(:,2);x3(:,2)]);
a2max = max([x1(:,2);x2(:,2);x3(:,2)]);

[a1 a2] = meshgrid(a1min:0.1:a1max, a2min:0.1:a2max);

testX=[a1(:) a2(:)];

% Experimenting with MLP

% number of epochs for training
nEpochs = 400;

% learning rate
eta = 0.01;

% number of hidden layer units
H = 16;
layers=1;
% train the MLP using the generated sample dataset
[w,h1,h2,h3 v, trainerror] = mlptrain([x1;x2;x3],[y1;y2;y3], H, eta, nEpochs,layers);

% plot the train error againt the number of epochs
figure; plot(1:nEpochs, trainerror, 'b:', 'LineWidth', 2);

ydash = mlptest(testX, w, h1,h2,h3, v);

[val idx] = max(ydash, [], 2);

label = reshape(idx, size(a1));

figure;
imagesc([a1min a1max], [a2min a2max], label), hold on,
set(gca, 'ydir', 'normal'),

cmap = [1 0.8 0.8; 0.9 1 0.9; 0.9 0.9 1];
colormap(cmap);

% plot the training data
plot(x1(:,1),x1(:,2),'r.', 'LineWidth', 2),
plot(x2(:,1),x2(:,2),'g+', 'LineWidth', 2),
plot(x3(:,1),x3(:,2),'bo', 'LineWidth', 2),

legend('Class 1', 'Class 2', 'Class 3', 'Location', 'NorthOutside', ...
    'Orientation', 'horizontal');


function [w h1 h2 h3 v  trainerror] = mlptrain(X, Y, H, eta, nEpochs,layers)
N = size(X,1);
D = size(X,2); % excluding the bias term
K = size(Y,2);

w = -0.3+(0.6)*rand(H,(D+1));
h1 =  -0.3+(0.6)*rand(H,(H+1));
h2 =  -0.3+(0.6)*rand(H,(H+1));
h3 =  -0.3+(0.6)*rand(H,(H+1));
v = -0.3+(0.6)*rand(K,(H+1));

iporder = randperm(N);
hidden_weight=[];
hidden_input=[];



% mlp training through stochastic gradient descent
for epoch = 1:nEpochs
    for n = 1:N
        x=X(iporder(n),:);%adding the bias term to training point X=300x2  tp=1x2
        y=Y(iporder(n),:);
        %x=X(n,:);
        %y=Y(n,:);
        x=x';
        y=y';
        
        x1=1./(1.+exp(-w*[1;x]));
        x2=1./(1.+exp(-h1*[1;x1]));
        x3=1./(1.+exp(-h2*[1;x2]));
        x4=1./(1.+exp(-h3*[1;x3]));
        x5=exp(v*[1;x4])/sum(exp(v*[1;x4]));
        
        delta4=(x5-y);
        dE5=delta4*[1;x4]';
        
        delta3 = (v(:,2:end)'*delta4).*(x4.*(1-x4));
        dE4 = delta3*[1;x3]';
        
        delta2 = (h3(:,2:end)'*delta3).*(x3.*(1-x3));
        dE3 = delta2*[1;x2]';

        delta1 = (h2(:,2:end)'*delta2).*(x2.*(1-x2));
        dE2=delta1*[1;x1]';
        
        delta0=(h1(:,2:end)'*delta1).*(x1.*(1-x1));
        dE1=delta0*[1;x]';
        
        v=v-eta*dE5;
        h3=h3-eta*dE4;
        h2=h2-eta*dE3;
        h1=h1-eta*dE2;
        w=w-eta*dE1;
        % 
        
    disp('#################');
    disp(y);
    disp(x5);
    disp('#################');
    end
    ydash = mlptest(X, w,h1,h2,h3,v);
     trainerror(epoch) =-sum(sum(Y.*log(ydash)));
    
end
%disp(trainerror);
return;

function ydash = mlptest(X, w,h1,h2,h3,v)
N = size(X,1);
K = size(v,1);
ydash=zeros(N,K);
for i=1:N
    x=X(i,:);
    x1=1./(1.+exp(-w*[1 x]'));
    x2=1./(1.+exp(-h1*[1;x1]));
    x3=1./(1.+exp(-h2*[1;x2]));
    x4=1./(1.+exp(-h3*[1;x3]));
    x5=exp(v*[1; x4])/sum(exp(v*[1 ;x4]));
    ydash(i,:)=x5';
end

return;
