%U[0,1] measure
%Functions we integrate lie in Sobolev space s=1
%Kernels we use in quadrature are Sobolev space t=1,2,3,4
%Lambda = 0

%Compare quadrature using different kernels in weights, and against
%classical methods (Simpson, Gauss-Legendre, Sobol)

tic
figure(1); clf; figure(2); clf
v = 200; s=1; itr = 1000; % v is no. x points. itr is no. iterations.
%s=1 corresponds to space containing our test functions
inc = 1; %increment between values of n

load kernels
%Loads vec, K(:,:,:) array of kernel matrices
%vec is vector of 2000 iid points from U[0,1]
%K(:,:,X) is the X-th kernel matrix evaluated on vec

%Ignore the above - I had trouble computing kernel matrices because of slow
%Bernoulli function, so I computed them and saved them for later use.
%But the kernel matrices can actually be computed relatively quickly since
%the early bernoulli polynomials are relatively simple/low degree, so this
%is what I recommend.

E = zeros(itr,v,4); E2 = E; %Arrays which will store our errors
for j=1:itr
    rp = randperm(2000);
    x = vec(rp(1:v)); %200 random points from our 2000
    %Creating a test function
    r=500;
    y = rand(r,1);
    c = randn(r,1);
    y1 = repmat(y,1,r);
    K_s = 1 + (((2*pi)^2)/(2*factorial(2)))*(abs(y1'-y1).^2-abs(y1'-y1)+(1/6));
    %Last part of this is a simple expression for bernoulli poly of degree 2.
    %i.e. x^2-x+1/6
    c = c/sqrt(c'*K_s*c); %Normalise function wrt RKHS norm
    h = zeros(r,v); %Matrix - (i,j)-th entry will be c_i*K(yi,xj)
    for m=1:v 
        h(:,m) = c.*(1 + (((2*pi)^2)/(2*factorial(2*s)))*((abs(x(m)*ones(r,1)-y)).^2-(abs(x(m)*ones(r,1)-y))+(1/6)));
    end
    h = sum(h,1); %Vector h(x(i)) = sum c_i*K(yi,xj)
    I = sum(c); %Integral of h
    toc %Approx 20 secs
    
    for t=1:4
        for n=1:inc:v
            b = K(rp(1:n),rp(1:n),t)\ones(n,1); %Kernel submatrix corresponding to chosen points
            A = h(1:n)*b; %Quadrature rule
            e = (A-I)^2; %Squared error
            E(j,n,t) = e; %Write sq err into array - K(:,:,X) stores errors from X-th kernel method
            if t==1
                E2(j,n,1) = e; %E compares quadrature using different kernels;
                %E2 compares kernel t=1/legendre/simpson/sobol.
            end
        end
    end
    toc
    
    %GL, Sobol, Simpson
    for n = 1:inc:v
        %Sobol
        x = net(sobolset(1),n); %First n points in Sobol sequence
        h = zeros(r,v); %Evaluating h at points of Sobol seq
        for m=1:n
            h(:,m) = c.*(1 + (((2*pi)^2)/(2*factorial(2*s)))*((abs(x(m)*ones(r,1)-y)).^2-(abs(x(m)*ones(r,1)-y))+(1/6)));
        end
        h = sum(h,1); %Vector h(x(i))
        A_So = sum(h(1:n))/n; %Sobol is quasi-monte carlo method
        E2(j,n,4) = (A_So - I)^2; %Squared error, write into E2
        
        %Simpson
        if mod(n,2) == 0 || n==1 %Simpson only works for odd no. points and not for n=1
        else
            x = 0:1/(n-1):1; k = (n-1)/2; %n points; [0,1] split into k intervals
            h = zeros(r,v); %Evaluating h at chosen points
            for m=1:n
                h(:,m) = c.*(1 + (((2*pi)^2)/(2*factorial(2*s)))*((abs(x(m)*ones(r,1)-y)).^2-(abs(x(m)*ones(r,1)-y))+(1/6)));
            end
            h = sum(h,1); %Vector h(x(i))
            M = zeros(k,3); %Implementing Simpson's method on each of k intervals
            for i=1:k
                M(i,:) = [h(2*i-1), 4*h(2*i), h(2*i+1)];
            end
            A_Si = sum(sum(M,1),2)/(6*k);
            E2(j,n,3) = (A_Si - I)^2;
        end
        
        %Gauss-Legendre
        %Up until 'w=2*V(.... ' this is code from elsewhere. It calculates
        %the abscissas and weights used in Gauss-Legendre.        
        
        i   = 1:n-1;
        a   = i./sqrt(4*i.^2-1);
        CM  = diag(a,1) + diag(a,-1);
        [V, L]   = eig(CM);
        [e, ind] = sort(diag(L));
        e = 0.5*(e+1); %Abscissas
        V       = V(:,ind)';
        w       = 2 * V(:,1).^2; %Weights
        h = zeros(r,v); %Evaluate h
        for m=1:n
            h(:,m) = c.*(1 + (((2*pi)^2)/(2*factorial(2*s)))*((abs(e(m)*ones(r,1)-y)).^2-(abs(e(m)*ones(r,1)-y))+(1/6)));
        end
        h = sum(h,1); %Vector h(x(i))
        
        %Implementing GL. Factor of 0.5 since standard GL is for functions on [-1,1] 
        A_GL = 0.5*h(1:n)*w;
        E2(j,n,2) = (A_GL - I)^2;
    end
end

%log10 of sqrt(avg(squared errors)))
AvE = 0.5*log10(sum(E,1)/itr); % 1 x n x 4 array
AvE2 = 0.5*log10(sum(E2,1)/itr); % 1 x n x 4 array

figure(1)
X = [ones(size([41:inc:v]')) log10([41:inc:v]')];
x = 0:2.5/50:2.5;
%X and x used for regression
for t=1:4 %Plotting errors and regression for 4 kernel methods
    if t==1
        coeff1 = regress(AvE(1,41:inc:v,t)',X); %Regression coeffs.
        y = coeff1'*[ones(size(x)); x];
        plot(log10([1:inc:v]),AvE(1,1:inc:v,t),'b.-'), hold on %Plot errors
        plot(x,y,'b:'), hold on % Plot regression
    elseif t==2
        coeff2 = regress(AvE(1,41:inc:v,t)',X);
        y = coeff2'*[ones(size(x)); x];
        plot(log10([1:inc:v]),AvE(1,1:inc:v,t),'g.-'), hold on
        plot(x,y,'g:'), hold on
    elseif t==3
        coeff3 = regress(AvE(1,41:inc:v,t)',X);
        y = coeff3'*[ones(size(x)); x];
        plot(log10([1:inc:v]),AvE(1,1:inc:v,t),'r.-'), hold on
        plot(x,y,'r:'), hold on
    elseif t==4
        coeff4 = regress(AvE(1,41:inc:v,t)',X);
        y = coeff4'*[ones(size(x)); x];
        plot(log10([1:inc:v]),AvE(1,1:inc:v,t),'c.-'), hold on
        plot(x,y,'c:'), hold on
    end
end
legend('t=1',num2str(-coeff1(2)),'t=2',num2str(-coeff2(2)),'t=3',num2str(-coeff3(2)),'t=4',num2str(-coeff4(2)))
toc

figure(2)
for t=1:4 %Errors and regression for kernel t=1/GL/Simpson/Sobol
    if t==1
        coeff12 = regress(AvE2(1,41:inc:v,t)',X);
        y = coeff12'*[ones(size(x)); x];
        plot(log10([1:inc:v]),AvE2(1,1:inc:v,t),'b.-'), hold on
        plot(x,y,'b:'), hold on
    elseif t==2
        coeff22 = regress(AvE2(1,41:inc:v,t)',X);
        y = coeff22'*[ones(size(x)); x];
        plot(log10(1:inc:v),AvE2(1,1:inc:v,t),'g.-'), hold on
        plot(x,y,'g:'), hold on
    elseif t==3
        if mod(inc,2)==0 %Since Simpson only work for odd n
            inc_ = inc;
        else inc_ = 2*inc;
        end
        coeff32 = regress(AvE2(1,41:inc_:v,t)',[ones(size([41:inc_:v]')) log10([41:inc_:v]')]);        y = coeff32'*[ones(size(x)); x];
        plot(log10(1:inc_:v),AvE2(1,1:inc_:v,t),'r.-'), hold on
        plot(x,y,'r:'), hold on
    elseif t==4
        coeff42 = regress(AvE2(1,41:inc:v,t)',X);
        y = coeff42'*[ones(size(x)); x];
        plot(log10(1:inc:v),AvE2(1,1:inc:v,t),'c.-'), hold on
        plot(x,y,'c:'), hold on
    end
end
legend('Kernel t=1',num2str(-coeff12(2)),'Legendre',num2str(-coeff22(2)),'Simpson',num2str(-coeff32(2)),'Sobol',num2str(-coeff42(2)))
toc

figure(1); savefig('kernels')
figure(2); savefig('comparison')