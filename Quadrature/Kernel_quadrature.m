%Quadrature

tic
figure(1); clf; figure(2); clf
v = 200; s=1; itr = 1; % v is no. x points. itr is no. iterations.
inc = 20; %increment between values of n

E = zeros(1000,v,4); E2 = E;
for j=1:itr
    x = rand(v,1);
    x1 = repmat(x,1,v);
    
    r=100; %Normally 500
    y = rand(r,1);
    c = randn(r,1);
    y1 = repmat(y,1,r);
    K_s = 1 + (((-1)^(s-1))*((2*pi)^(2*s))/(2*factorial(2*s)))*bernoulli(2*s,abs(y1'-y1)-floor(abs(y1'-y1)));
    c = c/sqrt(c'*K_s*c);
    h = zeros(r,v);
    for m=1:v 
        h(:,m) = c.*(1 + (((-1)^(s-1))*((2*pi)^(2*s))/(2*factorial(2*s)))*bernoulli(2,abs(x(m)*ones(r,1)-y)-floor(abs(x(m)*ones(r,1)-y))));
    end
    h = sum(h,1); %Vector h(x(i))
    I = sum(c); %Integral of h
    toc %Approx 20 secs
    
    K_1 = 1 + (((-1)^(1-1))*((2*pi)^(2*1))/(2*factorial(2*1)))*bernoulli(2*1,abs(x1'-x1)-floor(abs(x1'-x1)));
    K_2 = 1 + (((-1)^(2-1))*((2*pi)^(2*2))/(2*factorial(2*2)))*bernoulli(2*2,abs(x1'-x1)-floor(abs(x1'-x1)));
    K_3 = 1 + (((-1)^(3-1))*((2*pi)^(2*3))/(2*factorial(2*3)))*bernoulli(2*3,abs(x1'-x1)-floor(abs(x1'-x1)));
    K_4 = 1 + (((-1)^(4-1))*((2*pi)^(2*4))/(2*factorial(2*4)))*bernoulli(2*4,abs(x1'-x1)-floor(abs(x1'-x1)));
    toc %This step is FAR slower than everything else.
    
    for t=1:4
        for n=1:inc:v
            if t==1, K = K_1(1:n,1:n);
            elseif t==2, K = K_2(1:n,1:n);
            elseif t==3, K = K_3(1:n,1:n);
            elseif t==4, K = K_4(1:n,1:n);
            end
            b = K\ones(n,1);
            A = h(1:n)*b;
            e = (A-I)^2;
            E(j,n,t) = e;
            if t==1, E2(j,n,1) = e;
            else
            end
        end
    end
    toc
    
    %GL, Sobol, Simpson
    for n = 1:inc:v
        %Sobol
        x = net(sobolset(1),n);
        m=1; h = zeros(r,v);
        while m<=n, h(:,m) = c.*(1 + (((-1)^(s-1))*((2*pi)^(2*s))/(2*factorial(2*s)))*bernoulli(2,abs(x(m)*ones(r,1)-y)-floor(abs(x(m)*ones(r,1)-y)))); m=m+1;
        end
        h = sum(h,1); %Vector h(x(i))
        A_So = sum(h(1:n))/n;
        E2(j,n,4) = (A_So - I)^2;
        
        %Simpson
        if mod(n,2) == 0 || n==1
        else
            x = 0:1/(n-1):1; k = (n-1)/2;
            m=1; h = zeros(r,v);
            while m<=n, h(:,m) = c.*(1 + (((-1)^(s-1))*((2*pi)^(2*s))/(2*factorial(2*s)))*bernoulli(2,abs(x(m)*ones(r,1)-y)-floor(abs(x(m)*ones(r,1)-y)))); m=m+1;
            end
            h = sum(h,1); %Vector h(x(i))
            M = zeros(k,3);
            for i=1:k
                M(i,:) = [h(2*i-1), 4*h(2*i), h(2*i+1)];
            end
            A_Si = sum(sum(M,1),2)/(6*k);
            E2(j,n,3) = (A_Si - I)^2;
        end
        
        %Gauss-Legendre
        i   = 1:n-1;
        a   = i./sqrt(4*i.^2-1);
        CM  = diag(a,1) + diag(a,-1);
        [V, L]   = eig(CM);
        [e, ind] = sort(diag(L));
        e = 0.5*(e+1); %Abscissas
        V       = V(:,ind)';
        w       = 2 * V(:,1).^2; %Weights
        m=1; h = zeros(r,v);
        while m<=n, h(:,m) = c.*(1 + (((-1)^(s-1))*((2*pi)^(2*s))/(2*factorial(2*s)))*bernoulli(2,abs(e(m)*ones(r,1)-y)-floor(abs(e(m)*ones(r,1)-y)))); m=m+1;
        end
        h = sum(h,1); %Vector h(x(i))

        A_GL = 0.5*h(1:n)*w;
        E2(j,n,2) = (A_GL - I)^2;
    end
end
        
        
AvE = 0.5*log10(sum(E(1:j,:,:),1)/j); % 1 x n x 4 array
AvE2 = 0.5*log10(sum(E2(1:j,:,:),1)/j); % 1 x n x 4 array

figure(1)
X = [ones(size([21:inc:v]')) log10([21:inc:v]')];
x = 0:2.5/50:2.5;
for t=1:4
    if t==1
        coeff1 = regress(AvE(1,21:inc:v,t)',X);
        y = coeff1'*[ones(size(x)); x];
        plot(log10([1:inc:v]),AvE(1,1:inc:v,t),'bo-'), hold on
        plot(x,y,'b:'), hold on
    elseif t==2
        coeff2 = regress(AvE(1,21:inc:v,t)',X);
        y = coeff2'*[ones(size(x)); x];
        plot(log10([1:inc:v]),AvE(1,1:inc:v,t),'go-'), hold on
        plot(x,y,'g:'), hold on
    elseif t==3
        coeff3 = regress(AvE(1,21:inc:v,t)',X);
        y = coeff3'*[ones(size(x)); x];
        plot(log10([1:inc:v]),AvE(1,1:inc:v,t),'ro-'), hold on
        plot(x,y,'r:'), hold on
    elseif t==4
        coeff4 = regress(AvE(1,21:inc:v,t)',X);
        y = coeff4'*[ones(size(x)); x];
        plot(log10([1:inc:v]),AvE(1,1:inc:v,t),'co-'), hold on
        plot(x,y,'c:'), hold on
    end
end
legend('t=1',num2str(-coeff1(2)),'t=2',num2str(-coeff2(2)),'t=3',num2str(-coeff3(2)),'t=4',num2str(-coeff4(2)))
%How to get coeffs in legend?
toc

figure(2)
for t=1:4
    if t==1
        coeff12 = regress(AvE2(1,21:inc:v,t)',X);
        y = coeff12'*[ones(size(x)); x];
        plot(log10([1:inc:v]),AvE2(1,1:inc:v,t),'bo-'), hold on
        plot(x,y,'b:'), hold on
    elseif t==2
        coeff22 = regress(AvE2(1,21:inc:v,t)',X);
        y = coeff22'*[ones(size(x)); x];
        plot(log10(1:inc:v),AvE2(1,1:inc:v,t),'go-'), hold on
        plot(x,y,'g:'), hold on
    elseif t==3
        if mod(inc,2)==0
            inc_ = inc;
        else inc_ = 2*inc;
        end
        coeff32 = regress(AvE2(1,21:inc_:v,t)',[ones(size([21:inc_:v]')) log10([21:inc_:v]')]);        y = coeff32'*[ones(size(x)); x];
        plot(log10(1:inc_:v),AvE2(1,1:inc_:v,t),'ro-'), hold on
        plot(x,y,'r:'), hold on
    elseif t==4
        coeff42 = regress(AvE2(1,21:inc:v,t)',X);
        y = coeff42'*[ones(size(x)); x];
        plot(log10(1:inc:v),AvE2(1,1:inc:v,t),'co-'), hold on
        plot(x,y,'c:'), hold on
    end
end
legend('Kernel t=1',num2str(-coeff12(2)),'Legendre',num2str(-coeff22(2)),'Simpson',num2str(-coeff32(2)),'Sobol',num2str(-coeff42(2)))
%How to get coeffs in legend?
toc