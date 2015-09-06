%Brownian covariance kernel
%U[0,1] kernel
%Lambda non-zero (variable)

tic
figure(1); clf;
p = 1; %Dimension - fixed.
v = 500; itr = 150; % v is no. x points. itr is no. iterations.
inc = 1; %increment between values of n
lambda = 0.001; %error level

E = zeros(itr,200,5);%Array which will store our errors.
kernel = @(x,y)   ( min(x,y) )

brownian_optimal
%runs program giving: x1 is 10,001 points uniformly spread from 0 to 1
%with q (optimal dist.) evaluated at these points.

mass = zeros(1,2*length(q)-2);
mass(1) = q(1);
mass(end) = q(end);
for i = 2:length(q)-1
    mass(2*i-2) = q(i);
    mass(2*i-1) = q(i);
end
mass = mass/sum(mass);

%See 'brownian_piecewise_density.pdf' to see what mass is, and read the
%following:

%mass is a piecewise constant function on [0,1], constant on a
%bar around each point of x1, with bars of half width at the end
%points. We circumvent the problem of the end points by splitting every
%other interval in two, with the function taking the same value on both
%halves, hence the 'doubling' effect in the 'for' loop above.

w = 1/length(mass); %Distance between points where we have constructed 'mass'
Int = w*sum(mass); %Integral of piecewise constant function mass.
density = mass/Int; %Piecewise constant fn. normalised i.e. distribution


for j=1:itr %Each loop is approx ---- secs
    j
    
    x = rand(p,v); %U(0,1)
    
    %Creating a test function
    r=500;
    y = rand(p,r); %Each y is col vec length p.
    c = randn(1,r);
    K_y = zeros(r);
    for i=1:r
        K_y(i,:) = kernel(repmat(y(:,i),1,r),y);
    end
    c = c/sqrt(c*K_y*c'); %Normalise function wrt RKHS norm
    h = zeros(v,r); %Matrix - (i,j)-th entry will be c_j*K(xi,yj)
    for m=1:v
        h(m,:) = c.*kernel(repmat(x(:,m),1,r),y);
    end
    h = sum(h,2); %Column vector length v, h(x(i)) = sum c_j*K(xi,yj)
    
    %Mean embedding on y
    mu_y = 0.5*y.*(2-y); %Row vector length r

    I = mu_y*c'; %Integral of h
    
    %Kernel matrix for x:
    K_x = zeros(v);
    for i=1:v
        K_x(i,:) = kernel(repmat(x(:,i),1,v),x);
    end
    
    %Mean embedding on x:
    mu_x = 0.5*x.*(2-x); %Row vector length v
    
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
        %All adjustment by q is contained in our adjustment to K.
        A = h(rp(1:n))'*b; %Quadrature rule
        e = (A-I)^2; %Squared error
        E(j,n,1) = e; %Write sq err into array - E(:,:,t) stores errors from t-th method
        
        %Resampling method - lambda = 1
        s1 = datasample(1:v,n,'Replace',false,'Weights',q1); %Sample without replacement
        K_x2 = K_x(s1,s1) + n*lambda*diag(q1(s1));
        b = K_x2\mu_x(s1)';
        A = h(s1)'*b; %Quadrature rule
        e = (A-I)^2; %Squared error
        E(j,n,3) = e;
                
        %n-DPP method
        Y = sample_dpp(L,n);
        K_x2 = K_x(Y,Y) + n*lambda*diag(q1(Y)); %q = q1
        b = K_x2\mu_x(Y)';
        A = h(Y)'*b;
        e = (A-I)^2; %Squared error
        E(j,n,2) = e;
        
        %Resample - lambda = 0.01
        s2 = datasample(1:v,n,'Replace',false,'Weights',q2);
        K_x2 = K_x(s2,s2) + n*lambda*diag(q2(s2));
        b = K_x2\mu_x(s2)';
        A = h(s2)'*b; %Quadrature rule
        e = (A-I)^2; %Squared error
        E(j,n,4) = e;
        
        %Optimal
        s3 = datasample(1:length(mass),n,'Weights',mass); %WITH replacement
        x2 = (s3-1+rand(1,n))/length(mass); %Finding point on grid on [0,1] + noise
        K_x1 = kernel(repmat(x2,n,1),repmat(x2,n,1)')+n*lambda*diag(density(s3));
        
        %Here we made optimal density piecewise constant - draw a
        %sample with replacement from finitely many points where we know
        %'density', adjust with noise so we draw from the
        %[0,1] instead of our finite set of points (this is the
        %same as drawing from our piecewise constant density).
        
        %Note: We know optimal density on some points, then constructed
        %'density' earlier as the basis for our piecewise constant
        %distribution. We did this to circumvent problems near 0 and 1,
        %where the bars around these points are only of half width.
        
        mu_x1 = 0.5*x2.*(2-x2); %Row vector length n
        b = K_x1\mu_x1';
        h1 = zeros(r,n); %Matrix - (i,j)-th entry will be c_i*K(yi,x2j)
        for m=1:n
            h1(:,m) = c.*kernel(x2(m)*ones(1,r),y);
        end
        h1 = sum(h1,1); %Vector h1(x2(i)) = sum c_i*K(yi,x2j)
        A = h1*b;
        e = (A-I)^2;
        E(j,n,5) = e;    end
    toc
end

%log10 of sqrt(avg(squared errors)))
AvE = 0.5*log10(sum(E,1)/itr); % 1 x n x 5 array

figure(1)
X = [ones(size([41:inc:200]')) log10([41:inc:200]')];
x = 0:2.5/50:2.5;
%X and x used for regression
coeff = zeros(5,2); %Stores regression coeffs
str1 = ['b.-'; 'g.-'; 'r.-'; 'c.-'; 'm.-']; %For plotting errors
str2 = ['b:'; 'g:'; 'r:'; 'c:'; 'm:']; %For plotting regression
for t=[1 2 3 4 5] %Plotting errors and regression for 5 methods
    coeff(t,:) = regress(AvE(1,41:inc:200,t)',X); %Regression coeffs.
    y = coeff(t,:)*[ones(size(x)); x];
    plot(log10([1:inc:200]),AvE(1,1:inc:200,t),str1(t,:)), hold on %Plot errors
    plot(x,y,str2(t,:)), hold on % Plot regression
end

title(sprintf('Brownian - dimension:%3.0f - lambda %f',p,lambda))
legend('Bach',num2str(-coeff(1,2)),'n-DPP',num2str(-coeff(2,2)),'Resample 1',num2str(-coeff(3,2)),'Resample 0.01',num2str(-coeff(4,2)),'Optimal',num2str(-coeff(5,2)),'Location','SouthWest')
%legend('Bach',num2str(-coeff(1,2)),'Resample 1',num2str(-coeff(3,2)),'Resample 0.01',num2str(-coeff(4,2)),'Optimal',num2str(-coeff(5,2)),'Location','SouthWest')


toc

figure(1)
savefig('brownian_opt_lambda')