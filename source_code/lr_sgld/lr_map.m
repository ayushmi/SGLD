function [acc,elapsedTime] = lr_sgld(bs)
numIters = 5000;
burnin = round(0.2 * numIters);
thin_interval = round(0.02 * numIters);
num_collected = 0;
load('a9a_mat');
[N D] = size(X);
trainFrac = 0.8;
numTrain = round(trainFrac*N);
numTest = N - numTrain;
rp = randperm(N);
Xtrain = X(rp(1:numTrain),:);
Ytrain = Y(rp(1:numTrain));
Xtest = X(rp(numTrain+1:end),:);
Ytest = Y(rp(numTrain+1:end));
clear X;
clear Y;
w = randn(D,1);
wSaved = zeros(numIters-burnin,D);
%bsize = numTrain;
bsize = 10;%round(numTrain*bs);%100;
gamma = 0.55;
a = 0.01;b = 2.3;%100;
acc = [];
avg_prob = 0;
alpha = 10;

initacc = mean(sign(Xtest*w)==Ytest);
tot_time = 0;
burninTime = 0;


for t=1:numIters
    %ssize = 0.05;
    ssize = a*(b+t)^(-gamma);
    rp = randperm(numTrain);
    X = Xtrain(rp(1:bsize),:);
    Y = Ytrain(rp(1:bsize));
    %X = Xtrain(rp((t-1)*bsize+1:t*bsize),:);
    %Y = Ytrain(rp((t-1)*bsize+1:t*bsize));
    tic;
    p = sigmoid(-Y.*(X*w));
    grad_lik = X'*(Y.*p);
    grad_prior = -w/alpha; % assumuing a gaussian prior
    grad =  grad_prior + (numTrain/bsize)*grad_lik;
    w = w + 0.5*ssize*grad;% + sqrt(ssize)*randn(D,1);
    
    prob = sigmoid(Xtest*w);
    if t > burnin
        if mod(t,thin_interval) == 0  
            num_collected = num_collected + 1;
            avg_prob = (avg_prob*(num_collected-1) + prob)/num_collected;
            %pred = 2*(avg_prob>0.5)-1;
            pred = 2*(prob>0.5)-1;
            acc(num_collected) = mean(pred==Ytest);
            elapsedTime(num_collected) = toc;
            %plot(1:num_collected,acc);    
            %drawnow;
            fprintf('Post-burning phase: iteration = %d, accuracy (averaged over posterior)= %f\n',t,acc(num_collected))            
        end
        % save current sample
        wSaved(t-burnin,:) = w;
    else
            burninTime = burninTime + toc;
            pred = 2*(prob>0.5)-1;
            acc_pre_burnin = mean(pred==Ytest); 
            fprintf('Pre-burning phase: iteration = %d, (Unaveraged) accuracy = %f\n',t,acc_pre_burnin);
    end
    %elapsedTime = toc;
    %tot_time = tot_time + elapsedTime;
end
elapsedTime(1) = elapsedTime(1);%+burninTime;
for i=2:length(elapsedTime)
    elapsedTime(i)=elapsedTime(i-1)+elapsedTime(i);
end
%tot_time
end
