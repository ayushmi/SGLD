%function [finalTheta1,finalTheta2]= synthetic()
function [theta_1s,theta_2s]= synthetic()
	rng (5373);
	iterations = 10000;
	pts = 100;
	ptsPerBatch = 10;
	
	sigma_x = sqrt(2);
	sigma_1 = sqrt(10);
	sigma_2 = sqrt(1);
	theta_1 = normrnd(0,1);
	theta_2 = normrnd(0,1);

	for pt=1:pts
		if (rand() > 0.5)
			X(pt) = normrnd(0,sigma_x,1,1);
		else 
			X(pt) = normrnd(1,sigma_x,1,1);
		end
	end
	hist(X,100);
	T = 1:iterations;
	a = 0.2;
	b = 2.3;
	gamma = 0.55;

	theta_1s = [];
	theta_2s = [];

	for t = T
		disp(t)
		eta = a * (b + t)^(-gamma);
		lgrad1 = 0.0;
		lgrad2 = 0.0;
		indexes = randperm(pts);
		curBatch = X(indexes(1:ptsPerBatch));
		for x=curBatch
			[lgrad1,lgrad2] = analyticGrad(x,theta_1,theta_2,sigma_x);
			theta_1 = theta_1 +  eta * (pgrad(theta_1, 0, sigma_1) + (pts/ptsPerBatch)*lgrad1)/2.0 + normrnd(0,1)*sqrt(eta);
			theta_2 = theta_2 +  eta * (pgrad(theta_2, 0, sigma_2) + (pts/ptsPerBatch)*lgrad2)/2.0 + normrnd(0,1)*sqrt(eta);
			theta_1s = [theta_1s;theta_1];
			theta_2s = [theta_2s;theta_2];
		end
	end
	scatterhist(theta_1s,theta_2s);
end

function [gradTheta1,gradTheta2]= analyticGrad(x,theta_1,theta_2,sigma_x)
	likelihood1 = 0.5/sqrt(2*sigma_x^2*pi) * exp(-((x-theta_1)^2)/(2*sigma_x^2));
	likelihood2 = 0.5/sqrt(2*sigma_x^2*pi) * exp(-((x-theta_1-theta_2)^2)/(2*sigma_x^2)); 
	likelihood = likelihood1+likelihood2;
	dlogtheta1 = 2 * likelihood1* (-theta_1+x)/(2*sigma_x^2);;
	dlogtheta2 = 2 * likelihood2 * (-theta_1-theta_2+x)/(2*sigma_x^2);;
	gradTheta2 = (dlogtheta2)/likelihood; 
	gradTheta1 = (dlogtheta1+dlogtheta2)/likelihood; 
end

function output = pgrad(theta, mu, sigma)
	output = -theta/sigma^2;
end
