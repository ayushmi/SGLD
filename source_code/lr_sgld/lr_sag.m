function [acc,elapsedTime] = lr_sgld(bs)
numIters = 2000;
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
bsize = 10;
%bsize = 10;%round(numTrain*bs);%100;
gamma = 0.55;
a = 0.1;b = 2.3;%100;
acc = [];
avg_prob = 0;
alpha = 10;

initacc = mean(sign(Xtest*w)==Ytest);
tot_time = 0;

yis = zeros(numTrain,D); 
prev_sum=0;


for i=1:numTrain
    p = sigmoid(-Ytrain(i).*(Xtrain(i)*w));
    yis(i,:) = Xtrain(i)*(Ytrain(i).*p);
    prev_sum=prev_sum+sum(yis);
end
prev_sum=prev_sum/numTrain; 


for t=1:numIters
    ssize = 0.05;
    %ssize = a*(b+t)^(-gamma);
    rp = randperm(numTrain);

    tic;
    i=rp(1);
    p = sigmoid(-Ytrain(rp(1:bsize)).*(Xtrain(rp(1:bsize),:)*w));
    prev_sum = (numTrain*prev_sum - sum(yis(rp(1:bsize),:)))/numTrain;
    yis(rp(1:bsize),:) = repmat((Xtrain(rp(1:bsize),:)'*(Ytrain(rp(1:bsize)).*p)),1,bsize)';
    prev_sum = (numTrain*prev_sum + sum(yis(rp(1:bsize),:)))/numTrain;

    grad_prior = -w/alpha; % assumuing a gaussian prior
    grad =  grad_prior + prev_sum';
    w = w + 0.5*ssize*grad + sqrt(ssize)*randn(D,1);


    prob = sigmoid(Xtest*w);
    if t > burnin
        if mod(t,thin_interval) == 0  
            num_collected = num_collected + 1;
            avg_prob = (avg_prob*(num_collected-1) + prob)/num_collected;
            pred = 2*(avg_prob>0.5)-1;
            acc(num_collected) = mean(pred==Ytest);
            elapsedTime(num_collected) = toc;
            %plot(1:num_collected,acc);    
            %drawnow;
            fprintf('Post-burning phase: iteration = %d, accuracy (averaged over posterior)= %f\n',t,acc(num_collected))            
        end
        % save current sample
        wSaved(t-burnin,:) = w;
    else
            pred = 2*(prob>0.5)-1;
            acc_pre_burnin = mean(pred==Ytest); 
            fprintf('Pre-burning phase: iteration = %d, (Unaveraged) accuracy = %f\n',t,acc_pre_burnin);
    end
    %elapsedTime = toc;
    %tot_time = tot_time + elapsedTime;
end

for i=2:length(elapsedTime)
    elapsedTime(i)=elapsedTime(i-1)+elapsedTime(i);
end
%tot_time
end
