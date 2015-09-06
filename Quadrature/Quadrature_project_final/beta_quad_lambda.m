%Beta(0.5,0.5) measure
%Sobolev space s=1 kernel
%Lambda non-zero (variable)

tic

%Variables
v = 500; % v is no. x points.
itr = 5; %itr is no. iterations.
inc = 5; %increment between values of n
N = 5000; %detail of grid estimating mean embedding
lambda = 0.01; %Can vary
%We can also set lambda to zero and ignore the resampling method

E = zeros(itr,200,3); %Stores errors. E(j,n,method)
X = betarnd(0.5*ones(N,1),0.5); %For estimating mean embedding
a = .5; %Beta parameter

kernel = @(x,y)   ( 1 + (((2*pi)^2)/(2*factorial(2)))*(abs(x-y).^2-abs(x-y)+(1/6)) )
%Last part of this is a simple expression for bernoulli poly of degree 2.
%i.e. x^2-x+1/6

for j=1:itr %Each loop is approx ---- secs
    j
    x = betarnd(a,a,1,v); %500 random betas, row vector
    x1 = repmat(x,v,1); %Copy x into a matrix with identical rows.
    %Creating a test function
    r=500;
    y = rand(r,1);
    c = randn(r,1); %function = sum c.K(y,.)
    y1 = repmat(y,1,r);
    K_y = kernel(y1',y1); %Kernel matrix
    c = c/sqrt(c'*K_y*c); %Normalise function wrt RKHS norm
    h = zeros(r,v); %Matrix - (i,j)-th entry will be c_i*K(yi,xj)
    for m=1:v
        h(:,m) = c.*kernel(x(m)*ones(r,1),y); %m-th column is components of h(x(m))
    end
    h = sum(h,1); %Vector h(x(i)) = sum c_i*K(yi,xj)
    
    
    %Estimate mean embedding on y, in order to integrate h:
    mu_y = zeros(N,r); %Matrix - (i,j)-th entry will be K(Xi,yj)
    for m=1:r
        mu_y(:,m) = kernel(y(m)*ones(N,1),X);
    end
    mu_y = sum(mu_y,1)/N; %Vector mu(y(i)) = sum K(Xi,yj)/N = estimate of mean embedding at yi
    
    I = mu_y*c; %Approximate integral of h - HOW LARGE DO WE NEED N ?
    
    %Kernel matrix for x, s=1:
    K_x = kernel(x1',x1);
    
    q = 1; %Importance sampling density - don't know optimal in this case
    
    %Estimate mean embedding on x:
    mu_x = zeros(N,v); %Matrix - (i,j)-th entry will be K(Xi,xj)
    for m=1:v
        mu_x(:,m) = kernel(x(m)*ones(N,1),X);
    end
    mu_x = sum(mu_x,1)/N; %Vector mu(x(i)) = sum K(Xi,xj)/N = estimate of mean embedding at xi
    
    K1 = K_x*inv(K_x+lambda*eye(v)); %For resampling + n-DPP methods
    q1 = diag(K1)/trace(K1);
    
    %K2 = K_x*inv(K_x+eye(v));
    %q2 = diag(K2)/trace(K2); %Possible q for n-DPP case
    
    L = decompose_kernel(K_x); %n-DPP
    
    for n=1:inc:200 %No. points to use to perform quadrature
        %Monte Carlo with appropriate weights
        rp = randperm(v); %We will take n of our 500 random betas
        b = (K_x(rp(1:n),rp(1:n))+n*lambda*eye(n))\mu_x(rp(1:n))'; %Note q=1.
        A = h(rp(1:n))*b; %Quadrature rule
        e = (A-I)^2; %Squared error
        E(j,n,1) = e; %Write sq err into array - E(j,n,method)
        
        %n-DPP method
        Y = sample_dpp(L,n); %n-DPP - note: uses Kulesza/Taskar code
        b = (K_x(Y,Y)+n*lambda*diag(q1(Y)))\mu_x(Y)'; %Not sure about q=q1, not density??
        A = h(Y)*b; %Same as before
        e = (A-I)^2;
        E(j,n,2) = e;
        
        %Resampling method
        s1 = datasample(1:v,n,'Replace',false,'Weights',q1); %Sample without replacement according to weights q1
        b = (K_x(s1,s1)+n*lambda*diag(q1(s1)))\mu_x(s1)'; %q1 not density??
        A = h(s1)*b; %Same as before
        e = (A-I)^2;
        E(j,n,3) = e;
    end
end
toc

%log10 of sqrt(avg(squared errors)))
AvE = 0.5*log10(sum(E,1)/itr); % 1 x n x 3 array


coeff = zeros(3,2); %Regression coeffs
str1 = ['b.-'; 'g.-'; 'r.-']; %For plotting errors
str2 = ['b:'; 'g:'; 'r:']; %For plotting regression lines

figure(2);clf
for m = 1:3
    X = [ones(size([101:inc:200]')) log10([101:inc:200]')];
    x = 0:2.5/50:2.5;
    coeff(m,:) = regress(AvE(1,101:inc:200,m)',X); %Regression coeffs. for m-th method
    y = coeff(m,:)*[ones(size(x)); x]; %y=a+bx regression
    plot(log10(1:inc:200),AvE(1,1:inc:200,m),str1(m,:)), hold on %Plot errors
    plot(x,y,str2(m,:)), hold on % Plot regression
end

title(sprintf('Sobolev space s=1 / Beta(.5,.5) measure - lambda = %g',lambda))
legend('Beta',num2str(-coeff(1,2)),'n-DPP',num2str(-coeff(2,2)),'Resampling',num2str(-coeff(3,2)),'Location','SouthWest')
%savefig('beta_lambda_3') %Optional - save figure under some name