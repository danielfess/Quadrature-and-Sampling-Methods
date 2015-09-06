%Brownian covariance kernel
%U[0,1] kernel
%Lambda = 0

tic
figure(1); clf;
p = 1; %Dimension
v = 500; itr = 150; % v is no. x points. itr is no. iterations.
inc = 1; %increment between values of n

E = zeros(itr,200,4);%Array which will store our errors.
kernel = @(x,y)   ( min(x,y) )


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
    
    %         %Kernel matrix with columns adjusted by 1/sqrt(q(xj))
    %         %In this case, q = 1
    %         K_x1 = zeros(v);
    %         for col=1:v
    %             K_x1(:,col) = K_x(:,col)/sqrt(q(col)); %'col' short for 'column'
    %         end
    
    %Mean embedding on x:
    mu_x = 0.5*x.*(2-x); %Row vector length v
    
    K1 = K_x*inv(K_x+eye(v)); %For resampling method, lambda = 1
    q1 = diag(K1)/trace(K1);
    
    K2 = K_x*inv(K_x+0.01*eye(v)); %For resampling method, lambda = 0.01
    q2 = diag(K2)/trace(K2);
    
    L = decompose_kernel(K_x); %n-DPP
    
    for n=1:inc:200
        %Kernel method
        rp = randperm(v);
        b = K_x(rp(1:n),rp(1:n))\mu_x(rp(1:n))'; %Kernel submatrix corresponding to n points
        A = h(rp(1:n))'*b; %Quadrature rule
        e = (A-I)^2; %Squared error
        E(j,n,1) = e; %Write sq err into array - K(:,:,t) stores errors from t-th method
        
        %Resampling method - lambda = 1
        s1 = datasample(1:v,n,'Replace',false,'Weights',q1);
        K_x2 = zeros(v);
        for col=s1
            K_x2(:,col) = K_x(:,col)/sqrt(q1(col)); %'col' short for 'column'
        end
        %Note that since lambda=0, q does not need to appear in the
        %calculation of the weights, but I wrote this before I understood
        %this.
        b = K_x2(s1,s1)\mu_x(s1)'; %Kernel submatrix corresponding to n points
        A = h(s1)'*(b./sqrt(q1(s1))); %Quadrature rule
        e = (A-I)^2; %Squared error
        E(j,n,3) = e;
        
        %n-DPP method
        Y = sample_dpp(L,n);
        for col=Y'
            K_x2(:,col) = K_x(:,col)/sqrt(q1(col)); %'col' short for 'column'
        end
        b = K_x2(Y,Y)\mu_x(Y)';
        A = h(Y)'*(b./sqrt(q1(Y))); %q = q1 or q = 1?
        e = (A-I)^2; %Squared error
        E(j,n,2) = e;
        
        %Resample - lambda = 0.01
        s2 = datasample(1:v,n,'Replace',false,'Weights',q2);
        for col=s2
            K_x2(:,col) = K_x(:,col)/sqrt(q2(col)); %'col' short for 'column'
        end
        b = K_x2(s2,s2)\mu_x(s2)'; %Kernel submatrix corresponding to n points
        A = h(s2)'*(b./sqrt(q2(s2))); %Quadrature rule
        e = (A-I)^2; %Squared error
        E(j,n,4) = e;
        
    end
    toc
end

%log10 of sqrt(avg(squared errors)))
AvE = 0.5*log10(sum(E,1)/itr); % 1 x n x 4 array

figure(1)
X = [ones(size([41:inc:200]')) log10([41:inc:200]')];
x = 0:2.5/50:2.5;
%X and x used for regression
for t=1:4 %Plotting errors and regression for 4 methods
    if t==1
        coeff1 = regress(AvE(1,41:inc:200,t)',X); %Regression coeffs.
        y = coeff1'*[ones(size(x)); x];
        plot(log10([1:inc:200]),AvE(1,1:inc:200,t),'b.-'), hold on %Plot errors
        plot(x,y,'b:'), hold on % Plot regression
    elseif t==2
        coeff2 = regress(AvE(1,41:inc:200,t)',X);
        y = coeff2'*[ones(size(x)); x];
        plot(log10([1:inc:200]),AvE(1,1:inc:200,t),'g.-'), hold on
        plot(x,y,'g:'), hold on
    elseif t==3
        coeff3 = regress(AvE(1,41:inc:200,t)',X);
        y = coeff3'*[ones(size(x)); x];
        plot(log10([1:inc:200]),AvE(1,1:inc:200,t),'r.-'), hold on
        plot(x,y,'r:'), hold on
    elseif t==4
        coeff4 = regress(AvE(1,41:inc:200,t)',X);
        y = coeff4'*[ones(size(x)); x];
        plot(log10([1:inc:200]),AvE(1,1:inc:200,t),'c.-'), hold on
        plot(x,y,'c:'), hold on
    end
end
title(sprintf('Brownian - dimension:%3.0f',p))
legend('Kernel',num2str(-coeff1(2)),'n-DPP',num2str(-coeff2(2)),'Resample 1',num2str(-coeff3(2)),'Resample 0.01',num2str(-coeff4(2)),'Location','SouthWest')

toc

savefig(sprintf('brownian_dim%.0f',p))