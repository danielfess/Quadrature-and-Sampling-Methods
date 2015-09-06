%Gaussian measure on R
%SE kernel
%Lambda non-zero (variable)

tic
figure(1); clf;
v = 400; itr = 200; % v is no. x points. itr is no. iterations.
si2 = 2; %Kernel parameter
t2 = 100; %Distribution parameter
lambda = [1 0.1 0.01 0.001 0.0000001]; %Error level

E = zeros(itr,3,length(lambda));%Array which will store our errors.
kernel = @(x,y)   ( exp(-(x-y).^2/(2*si2)) )
mu = @(x)   ( (si2/(si2+t2))^0.5)*exp(-x.^2/(2*(si2+t2)) ) %Mean embedding
N = zeros(itr,length(lambda)); %Records N = d(lambda) for each j, lambda
N_DPP = N; %Records size of sets selected by DPP.

for t=1:length(lambda)
    for j=1:itr %Each loop is approx --- secs
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
        
        %Mean embedding on y
        mu_y = ((si2/(si2+t2))^0.5)*exp(-y.^2/(2*(si2+t2)));
        I = mu_y'*c; %Integral of h
        
        %Kernel matrix for x:
        K_x = kernel(xm',xm);
        
        %Mean embedding on x:
        mu_x = ((si2/(si2+t2))^0.5)*exp(-x.^2/(2*(si2+t2)));
        
        K1 = K_x*inv(K_x+lambda(t)*eye(v)); %To find n = d(lambda(t))
        tr = trace(K1);
        q1 = diag(K1)/tr;
        n = ceil(tr); %n = d(lambda(t)) (approx.)
        N(j,t) = n;
        
        L = decompose_kernel(K_x/lambda(t)); %DPP - have to decompose K/lambda
        %K isn't sufficient because  we are sampling from DPP as well as
        %n-DPP        
        
        %Monte Carlo with appropriate weights
        rp = randperm(v);
        b = (K_x(rp(1:n),rp(1:n))+n*lambda(t)*eye(n))\mu_x(rp(1:n))';
        %Kernel submatrix adjusted according to q, lambda(t).
        A = h(rp(1:n))*b; %Quadrature rule
        e = (A-I)^2; %Squared error
        E(j,1,t) = e; %Write sq err into array
        %E(j,r,t) - j is run, r is method, t is lambda
        
        %n-DPP
        Y = sample_dpp(L,n);
        K_x2 = K_x(Y,Y) + n*lambda(t)*diag(q1(Y)); %q = q1
        b = K_x2\mu_x(Y)';
        A = h(Y)*b;
        e = (A-I)^2; %Squared error
        E(j,2,t) = e;
        
        %DPP
        Y = sample_dpp(L); %Samples set (of any size) from DPP
        K_x2 = K_x(Y,Y) + length(Y)*lambda(t)*diag(q1(Y)); %q = q1
        b = K_x2\mu_x(Y)';
        A = h(Y)*b;
        e = (A-I)^2; %Squared error
        E(j,3,t) = e;
        N_DPP(j,t) = length(Y); %Size of DPP sample
    end
    toc
end

%Mean of errors, for fixed method and lambda
Mean = sum(E,1)/itr; % 1 x 3 x length(lambda) array

%Mean of n
n = N;
N = sum(N,1)/itr; %1 x length(lambda) matrix
%d(lambda) for each lambda

%log10 of sqrt(squared errors)
logE = 0.5*log10(E);

str = ['o'; '+'; '*'; '.'; 'x'; 's'; 'o'];
figure(1); clf
for t=1:length(lambda) %Plotting errors
    figure(t); clf
    scatter(log10(N(t))*ones(itr,1)+0.025*rand(itr,1),logE(:,1,t),15,'r',str(t)); hold on
    scatter(log10(N(t))*ones(itr,1)+0.025*rand(itr,1),logE(:,2,t),15,'g',str(t)); hold on
    scatter(log10(N_DPP(:,t))+0.025*rand(itr,1),logE(:,3,t),15,'b',str(t)); hold on
    %title(sprintf('Gaussian vs n-DPP vs DPP - lambda: %g',lambda(t)))
    %legend('Gaussian','n-DPP','DPP')
end
toc

%Note: script unfinished. Addition of error bars and perhaps some polishing
%required.