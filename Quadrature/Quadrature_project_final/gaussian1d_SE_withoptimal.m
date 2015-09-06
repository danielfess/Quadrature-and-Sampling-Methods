%Gaussian measure on R
%SE kernel
%Lambda non-zero (variable)

tic
figure(1); clf;
v = 500; itr = 10; % v is no. x points. itr is no. iterations.
si2 = 2; %Kernel parameter - length squared
t2 = 1; %Distribution parameter - variance
lambda = 0.000001; %Error level
inc = 1; %increment between values of n

E = zeros(itr,200,5);%Array which will store our errors.
kernel = @(x,y)   ( exp(-(x-y).^2/(2*si2)) )

%Select as appropriate: run gaussian_optimal or load saved variables

%Option 1
Gaussian_optimal
%Calculates the optimal distribution on x1 = -5:1/200:5 for our chosen lambda

%Option 2
%load gaussian_optimal.mat
%Loads the optimal distribution on x1 = -5:1/200:5 - only when lambda = 0.0001.

mass = (density/sum(density))';
%density is optimal density wrt Lebesgue evaluated on x1
%mass is mass on x1 wrt Lebesgue (i.e. mass from q dp)

w = 10/(length(density)-1); %Distance between points where we know density
Int = w*sum(density); %Integral of piecewise constant function taking values of density
density = density/Int; %Piecewise constant fn. normalised i.e. distribution

for j=1:itr %Each loop is approx 70 secs
    j
    
    x = sqrt(t2)*randn(1,v); %N(0,tau2)
    xm = repmat(x,v,1); %xm = x matrix
    %Creating a test function
    r=500;
    y = rand(r,1);
    c = randn(r,1);
    y1 = repmat(y,1,r);
    K_y = kernel(y1',y1);
    c = c/sqrt(c'*K_y*c); %Normalise function wrt RKHS norm
    h = zeros(r,v); %Matrix - (i,j)-th entry will be c_i*K(yi,xj)
    for m=1:v
        h(:,m) = c.*kernel(x(m)*ones(r,1),y);
    end
    h = sum(h,1); %Vector h(x(i)) = sum c_i*K(yi,xj)
    
    %Mean embedding on y - we have exact formula - gaussian integral can be
    %calculated.
    mu_y = ((si2/(si2+t2))^0.5)*exp(-y.^2/(2*(si2+t2)));
    I = mu_y'*c; %Integral of h
    
    %Kernel matrix for x:
    K_x = kernel(xm',xm);
    
    %Mean embedding on x:
    mu_x = ((si2/(si2+t2))^0.5)*exp(-x.^2/(2*(si2+t2)));
    
    K1 = K_x*inv(K_x+eye(v)); %For resampling method, lambda = 1
    q1 = diag(K1)/trace(K1);
    
    K2 = K_x*inv(K_x+0.01*eye(v)); %For resampling method, lambda = 0.01
    q2 = diag(K2)/trace(K2);
    
     L = decompose_kernel(K_x); %n-DPP
    
    for n=1:inc:200
        %Monte Carlo with appropriate weights
        rp = randperm(v);
        b = (K_x(rp(1:n),rp(1:n))+n*lambda*eye(n))\mu_x(rp(1:n))';
        %Kernel submatrix adjusted according to q, lambda.
        A = h(rp(1:n))*b; %Quadrature rule
        e = (A-I)^2; %Squared error
        E(j,n,1) = e; %Write sq err into array - E(:,:,t) stores errors from t-th method
        
        %Resampling method - lambda = 1
        s1 = datasample(1:v,n,'Replace',false,'Weights',q1); %Sample without replacement according to mass q1
        K_x2 = K_x(s1,s1) + n*lambda*diag(q1(s1));
        b = K_x2\mu_x(s1)';
        %All adjustment by q is contained in our adjustment to K.
        A = h(s1)*b; %Quadrature rule
        e = (A-I)^2; %Squared error
        E(j,n,3) = e;
                
        %n-DPP method
        Y = sample_DPP(L,n);
        K_x2 = K_x(Y,Y) + n*lambda*diag(q1(Y)); %q = q1?
        b = K_x2\mu_x(Y)';
        A = h(Y)*b;
        e = (A-I)^2; %Squared error
        E(j,n,2) = e;
        
        %Resample - lambda = 0.01
        s2 = datasample(1:v,n,'Replace',false,'Weights',q2);
        K_x2 = K_x(s2,s2) + n*lambda*diag(q2(s2));
        b = K_x2\mu_x(s2)';
        A = h(s2)*b; %Quadrature rule
        e = (A-I)^2; %Squared error
        E(j,n,4) = e;
        
        %Optimal
        s3 = datasample(1:length(x1),n,'Weights',mass); %WITH replacement
        x2 = x1(s3)+w*(rand(1,n)-0.5); %+ noise
        K_x1 = kernel(repmat(x2,n,1),repmat(x2,n,1)')+n*lambda*diag(density(s3)./normpdf(x2,0,sqrt(t2)));
        
        %Here we made density (wrt lebesgue) piecewise constant: draw a
        %sample with replacement from finitely many points where we know
        %the optimal density exactly, adjust with noise so we draw from the
        %reals instead of our finite set of points (up to now this is the
        %same as drawing from a piecewise constant density). Density wrt
        %gaussian is given by 'lebesgue density'/normpdf
        %(alternatively, lebesgue density = q dp). So now we have our
        %density wrt gaussian on the reals (approximately).
        
        mu_x1 = ((si2/(si2+t2))^0.5)*exp(-x2.^2/(2*(si2+t2)));
        b = K_x1\mu_x1';
        h1 = zeros(r,n); %Matrix - (i,j)-th entry will be c_i*K(yi,x2j)
        for m=1:n
            h1(:,m) = c.*kernel(x2(m)*ones(r,1),y);
        end
        h1 = sum(h1,1); %Vector h1(x2(i)) = sum c_i*K(yi,x2j)
        A = h1*b;
        e = (A-I)^2;
        E(j,n,5) = e;
    end
    toc
end

%log10 of sqrt(avg(squared errors)))
AvE = 0.5*log10(sum(E,1)/itr); % 1 x n x 4 array

figure(1)
X = [ones(size([41:inc:200]')) log10([41:inc:200]')];
x = 0:2.5/50:2.5;
%X and x used for regression
coeff = zeros(5,2); %Stores regression coeffs
str1 = ['b.-'; 'g.-'; 'r.-'; 'c.-'; 'm.-']; %For plotting errors
str2 = ['b:'; 'g:'; 'r:'; 'c:'; 'm:']; %For plotting regression lines
for t=[1 2 3 4 5] %Plotting errors and regression for 5 methods
    coeff(t,:) = regress(AvE(1,41:inc:200,t)',X); %Regression coeffs.
    y = coeff(t,:)*[ones(size(x)); x];
    plot(log10([1:inc:200]),AvE(1,1:inc:200,t),str1(t,:)), hold on %Plot errors
    plot(x,y,str2(t,:)), hold on % Plot regression
end
title(sprintf('Gaussian 1d - lambda: %g',lambda))
legend('Bach',num2str(-coeff(1,2)),'n-DPP',num2str(-coeff(2,2)),'Resample 1',num2str(-coeff(3,2)),'Resample 0.01',num2str(-coeff(4,2)),'Optimal',num2str(-coeff(5,2)),'Location','SouthWest')

toc

% figure(1)
% savefig('Gaussian_dim1_optimal')
%Saving figure is optional