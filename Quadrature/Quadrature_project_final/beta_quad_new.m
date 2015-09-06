%Beta(0.5,0.5) measure
%Sobolev space s=1 kernel
%Lambda = 0

tic
%Variables
v = 500; s=1; itr = 40; % v is no. x points. itr is no. iterations.
%s=1 corresponds to space containing our test functions
N = 5000; %detail of grid estimating mean embedding
inc = 1; %increment between values of n

E = zeros(itr,200,6,5); %Stores errors. E(j,n,method,beta)
X = betarnd(0.5*ones(N,1),0.5); %For estimating mean embedding
a = [.25 .5 .75 1 2]; %Beta parameters

kernel = @(x,y)   ( 1 + (((2*pi)^2)/(2*factorial(2)))*(abs(x-y).^2-abs(x-y)+(1/6)) )
%Last part of this is a simple expression for bernoulli poly of degree 2.
%i.e. x^2-x+1/6

for j=1:itr %Each loop is approx 0.7 secs
    j
    for t=1:5
        x = betarnd(a(t),a(t),1,v); %Make sure script elements (e.g for loops) are in optimum order!!
        x1 = repmat(x,v,1);
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
        
    
        %Estimate mean embedding on y, in order to integrate h:
        mu_y = zeros(N,r); %Matrix - (i,j)-th entry will be K(Xi,yj)
        for m=1:r
            mu_y(:,m) = kernel(y(m)*ones(N,1),X);
        end
        mu_y = sum(mu_y,1)/N; %Vector mu(y(i)) = sum K(Xi,yj)/N = estimate of mean embedding at yi
    
        I = mu_y*c; %Approximate integral of h - HOW LARGE DO WE NEED N ?

        %Kernel matrix for x, s=1:
        K_x = kernel(x1',x1);
        
        %Kernel matrix with columns adjusted by 1/sqrt(q(xj))
        q = betapdf(x',a(t),a(t))./betapdf(x',.5,.5); %Density of B(a,a) wrt B(.5,.5)
        q_inv = betapdf(x',.5,.5)./betapdf(x',a(t),a(t));
        %Problems with .25 case here? Most points near 0 or 1, and here q
        %very large.
    
        %Estimate mean embedding on x:
        mu_x = zeros(N,v); %Matrix - (i,j)-th entry will be K(Xi,xj)
        for m=1:v
            mu_x(:,m) = kernel(x(m)*ones(N,1),X);
        end
        mu_x = sum(mu_x,1)/N; %Vector mu(x(i)) = sum K(Xi,xj)/N = estimate of mean embedding at xi
    
        K1 = K_x*inv(K_x+eye(v)); %For resampling method, lambda = 1
        q1 = diag(K1)/trace(K1);
        
        K2 = K_x*inv(K_x+0.01*eye(v)); %For resampling method, lambda = 0.01
        q2 = diag(K2)/trace(K2);
        
        L = decompose_kernel(K_x); %n-DPP
        
        for n=1:inc:200
            %Kernel method
            rp = randperm(v); %Are the points still distributed according to beta(a(t),a(t)) ?
            b = K_x(rp(1:n),rp(1:n))\mu_x(rp(1:n))'; %Weighting by q is hidden here.
            A = h(rp(1:n))*b; %Quadrature rule
            e = (A-I)^2; %Squared error
            E(j,n,1,t) = e; %Write sq err into array - K(:,:,method,t) stores errors from t-th beta
            
            %n-DPP method
            Y = sample_dpp(L,n);
            b = K_x(Y,Y)\mu_x(Y)';
            A = h(Y)*b;
            e = (A-I)^2; %Squared error
            E(j,n,2,t) = e;
            
            %Resampling method - lambda = 1
            s1 = datasample(1:v,n,'Replace',false,'Weights',q1);
            b = K_x(s1,s1)\mu_x(s1)'; %Kernel submatrix corresponding to n points
            A = h(s1)*b; %Quadrature rule
            e = (A-I)^2; %Squared error
            E(j,n,3,t) = e;
            
            %Resample - lambda = 0.01
            s2 = datasample(1:v,n,'Replace',false,'Weights',q2);
            b = K_x(s2,s2)\mu_x(s2)'; %Kernel submatrix corresponding to n points
            A = h(s2)*b; %Quadrature rule
            e = (A-I)^2; %Squared error
            E(j,n,4,t) = e;
            
            %Gauss-Legendre
            %Up until 'w=2*V(.... ' this is code from elsewhere. It calculates
            %the abscissas and weights used in Gauss-Legendre.
            
            i   = 1:n-1;
            A   = i./sqrt(4*i.^2-1);
            CM  = diag(A,1) + diag(A,-1);
            [VEC, VAL]   = eig(CM);
            [e, ind] = sort(diag(VAL));
            e = 0.5*(e+1); %Abscissas
            VEC       = VEC(:,ind)';
            w       = 2 * VEC(:,1).^2; %Weights
            h1 = zeros(r,n); %Evaluate h
            for m=1:n
                h1(:,m) = c.*kernel(e(m)*ones(r,1),y);
            end
            h1 = sum(h1,1).*betapdf(e',0.5,0.5); %Vector h(e(i))
            %Implementing GL. Factor of 0.5 since standard GL is for functions on [-1,1]
            A_GL = 0.5*h1(1:n)*w;
            E(j,n,5,t) = (A_GL - I)^2;
            
            
            %Simpson
            if mod(n,2) == 0 || n==1 %Simpson only works for odd no. points and not for n=1
            else
                x2 = 1/(n+1):1/(n+1):n/(n+1); k = (n-1)/2; %n points; [1/(n+1),1-1/(n+1)] split into k intervals
                %Note we cannot have points at 0,1 since here betapdf is
                %infinite.
                h2 = zeros(r,n); %Evaluating h at chosen points
                for m=1:n
                    h2(:,m) = c.*kernel(x2(m)*ones(r,1),y);
                end
                h2 = sum(h2,1).*betapdf(x2,0.5,0.5); %Vector h(x2(i))
                Matrix = zeros(k,3); %Implementing Simpson's method on each of k intervals
                for i=1:k
                    Matrix(i,:) = [h2(2*i-1), 4*h2(2*i), h2(2*i+1)];
                end
                A_Si = sum(sum(Matrix,1),2)/(6*k);
                E(j,n,6,t) = (A_Si - I)^2;
            end
        end
    end
    toc
end

%log10 of sqrt(avg(squared errors)))
AvE = 0.5*log10(sum(E,1)/itr); % 1 x n x 4 x 5 array


coeff = zeros(6,5,2); %Regression coeffs
str1 = ['b.-'; 'g.-'; 'r.-'; 'c.-'; 'm.-'; 'k.-']; %For plotting errors
str2 = ['b:'; 'g:'; 'r:'; 'c:'; 'm:'; 'k:']; %For plotting regression lines

for t=1:5 %Plotting errors and regression for each beta separately
    figure(t); clf;
    for m = [1 2 3 4 5 6]
        if m~=6
            inc_ = inc;
        else
            inc_ = 2*inc; %Simpson only works for odd n
        end
        X = [ones(size([101:inc_:200]')) log10([101:inc_:200]')];
        x = 0:2.5/50:2.5;
        coeff(m,t,:) = regress(AvE(1,101:inc_:200,m,t)',X); %Regression coeffs.
        y = squeeze(coeff(m,t,:))'*[ones(size(x)); x];
        plot(log10(1:inc_:200),AvE(1,1:inc_:200,m,t),str1(m,:)), hold on %Plot errors
        plot(x,y,str2(m,:)), hold on % Plot regression
    end
    title(sprintf('Log-log plot of errors for q = beta(%.2f,%.2f)',a(t),a(t)))
    legend('Bach',num2str(-coeff(1,t,2)),'n-DPP',num2str(-coeff(2,t,2)),'Resampling 1',num2str(-coeff(3,t,2)),'Resampling 0.01',num2str(-coeff(4,t,2)),'Legendre',num2str(-coeff(5,t,2)),'Simpson',num2str(-coeff(6,t,2)),'Location','SouthWest')
    %legend('Bach',num2str(-coeff(1,t,2)),'Resampling 1',num2str(-coeff(3,t,2)),'Resampling 0.01',num2str(-coeff(4,t,2)),'Legendre',num2str(-coeff(5,t,2)),'Simpson',num2str(-coeff(6,t,2)),'Location','SouthWest')

end

savefig(1:5,'beta_new2')
%Note: Figure 2 is where we use the methods detailed in my Lyx document
%The other figures are where we draw from a different distribution (not rho)
%then apply our different sampling methods. They can be ignored if you
%want. I have included them since they are based off of one of Bach's
%figures.