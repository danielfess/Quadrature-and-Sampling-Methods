%Gaussian quadrature - general dimension

tic
figure(1); clf;

%Variables
p = 3; %Dimension we are working in.
v = 500; itr = 5; % v is no. x points. itr is no. iterations.
inc = 5; %increment between values of n
si2 = 1; %Kernel parameter (sigma^2)
t2 = 2; %Distribution parameter (tau^2) - multivariate indep gaussian

E = zeros(itr,200,4);%Array which will store our errors. E(j,n,method)
kernel = @(x,y)   ( exp(-sum((x-y).^2,1)/(2*si2)) ) %Operates on columns


for j=1:itr %Each loop is approx ---- secs
    j
    
    x = sqrt(t2)*randn(p,v); %N(0,tau2). Each point is col vec length p.
    %Creating a test function
    r=500;
    y = rand(p,r); %Each point is col vec length p.
    c = randn(1,r); %function = sum c.K(y,.)
    K_y = zeros(r);
    for i=1:r
        K_y(i,:) = kernel(repmat(y(:,i),1,r),y); %K(y_i,y_1),...,K(y_i,y_r)
    end
    c = c/sqrt(c*K_y*c'); %Normalise function wrt RKHS norm
    h = zeros(v,r); %Matrix - (i,j)-th entry will be c_j*K(xi,yj)
    for m=1:v
        h(m,:) = c.*kernel(repmat(x(:,m),1,r),y);
    end
    h = sum(h,2); %Column vector length v, h(x(i)) = sum c_j*K(xi,yj)
    
    %Mean embedding on y
    mu_y = ((si2/(si2+t2))^(p/2))*exp(-sum(y.^2,1)/(2*(si2+t2))); %Row vector length r

    I = mu_y*c'; %Integral of h = sum mu(y_i)*c_i
    
    %Kernel matrix for x:
    K_x = zeros(v);
    for i=1:v
        K_x(i,:) = kernel(repmat(x(:,i),1,v),x);
    end
        
    %Mean embedding on x:
    mu_x = ((si2/(si2+t2))^(p/2))*exp(-sum(x.^2,1)/(2*(si2+t2))); %Row vector length v
    
    K1 = K_x*inv(K_x+eye(v)); %For resampling method, lambda = 1
    q1 = diag(K1)/trace(K1);
    
    K2 = K_x*inv(K_x+0.01*eye(v)); %For resampling method, lambda = 0.01
    q2 = diag(K2)/trace(K2);
    
    L = decompose_kernel(K_x); %DPP
    
    for n=1:inc:200
        %Kernel method
        rp = randperm(v);
        b = K_x(rp(1:n),rp(1:n))\mu_x(rp(1:n))'; %lambda=0 here, so no q
        A = h(rp(1:n))'*b; %Quadrature rule
        e = (A-I)^2; %Squared error
        E(j,n,1) = e; %Write sq err into array - K(:,:,t) stores errors from t-th method
        
        %Resampling method - lambda = 1
        %Note: this doesn't really mean anything because we have no
        %regularisation, but when I didn't understand what it meant I
        %included it, so I will keep it here since my figures include it.
        s1 = datasample(1:v,n,'Replace',false,'Weights',q1); %Sample n points according to mass q1 without replacement
        b = K_x(s1,s1)\mu_x(s1)'; %Kernel submatrix corresponding to n points, again no q
        A = h(s1)'*b; %Quadrature rule
        e = (A-I)^2; %Squared error
        E(j,n,3) = e;
        
        %DPP method
        Y = sample_dpp(L,n); %n-DPP
        b = K_x(Y,Y)\mu_x(Y)';
        A = h(Y)'*b;
        e = (A-I)^2;
        E(j,n,2) = e;
        
        %Resample - lambda = 0.01
        %Again, this doesn't mean much but I'll keep it in.
        s2 = datasample(1:v,n,'Replace',false,'Weights',q2);
        b = K_x(s2,s2)\mu_x(s2)';
        A = h(s2)'*b;
        e = (A-I)^2;
        E(j,n,4) = e;
        
    end
    toc
end

%log10 of sqrt(avg(squared errors)))
AvE = 0.5*log10(sum(E,1)/itr); % 1 x n x 4 array

figure(1)
X = [ones(size([41:inc:200]')) log10([41:inc:200]')]; %To find regression coeffs
x = 0:2.5/50:2.5; %To plot regression
coeff = zeros(4,2); %Regression coeffs
str1 = ['b.-'; 'g.-'; 'r.-'; 'c.-']; %For plotting errors
str2 = ['b:'; 'g:'; 'r:'; 'c:']; %For plotting regression lines

for m = 1:4
    coeff(m,:) = regress(AvE(1,41:inc:200,m)',X); %Regression coeffs. for m-th method
    y = coeff(m,:)*[ones(size(x)); x]; %y=a+bx regression
    plot(log10(1:inc:200),AvE(1,1:inc:200,m),str1(m,:)), hold on %Plot errors
    plot(x,y,str2(m,:)), hold on % Plot regression
end

title(sprintf('Gaussian - dimension: %.0f',p))
legend('Gaussian',num2str(-coeff(1,2)),'DPP',num2str(-coeff(2,2)),'Resample 1',num2str(-coeff(3,2)),'Resample 0.01',num2str(-coeff(4,2)),'Location','SouthWest')

toc

%savefig(sprintf('gaussian_dim%.0f',p)) %Optional