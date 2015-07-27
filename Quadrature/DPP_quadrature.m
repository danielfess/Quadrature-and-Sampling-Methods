%DPP quadrature

%Note: settings for figures:
%DPPkernel.fig - inc = 5, itr = 20
%DPPkernel2.fig - inc = 1, itr = 200

tic
figure(3); clf;
load kernels.mat
%Loads kernels and vec - vector of 2000 points
itr = 200; %No. runs
s=1; %Sobolev space of test functions
inc = 1; %Density of values of n
L = decompose_kernel(K(:,:,1)); %Decomposes Kernel matrix for t=1
E = zeros(1000,200,2); %Errors for DPP and Kernel methods
toc %3 secs

for j=1:itr
    
    %Construct function to integrate
    r=500;
    y = rand(r,1);
    c = randn(r,1);
    y1 = repmat(y,1,r);
    K_s = 1 + (((2*pi)^2)/(2*factorial(2)))*(abs(y1'-y1).^2-abs(y1'-y1)+(1/6));
    c = c/sqrt(c'*K_s*c);
    h = zeros(r,2000);
    for m=1:2000 
        h(:,m) = c.*(1 + (((2*pi)^2)/(2*factorial(2*s)))*((abs(vec(m)*ones(r,1)-y)).^2-(abs(vec(m)*ones(r,1)-y))+(1/6)));
    end
    h = sum(h,1); %Vector h(vec(i))
    I = sum(c); %Integral of h
    %toc tells us + <0.1 secs
    
    %DPP quadrature
    for n=1:inc:200
        Y = sample_dpp(L,n); %Samples a set of size n from the DPP (equivalently: samples from the n-dpp)
        %sample_dpp takes a long time
        b = K(Y,Y,1)\ones(n,1); %Kernel submatrix corresponding to chosen points
        A = h(Y)*b;
        e = (A-I)^2;
        E(j,n,2) = e;
    end
    toc
    
    %Kernel quadrature (t=1)
    rp = randperm(2000);
    for n=1:inc:200
        b = K(rp(1:n),rp(1:n),1)\ones(n,1); %Kernel submatrix corresponding to chosen points
        A = h(rp(1:n))*b;
        e = (A-I)^2;
        E(j,n,1) = e;
    end
    toc %+ <0.1 secs
    
end

AvE = 0.5*log10(sum(E(1:j,:,:),1)/j); % 1 x n x 2 array

%Plotting errors and regression
figure(3)
X = [ones(size([41:inc:200]')) log10([41:inc:200]')];
x = 0:2.5/50:2.5;
for t=1:4
    if t==1
        coeff1 = regress(AvE(1,41:inc:200,t)',X);
        y = coeff1'*[ones(size(x)); x];
        plot(log10([1:inc:200]),AvE(1,1:inc:200,t),'b.-'), hold on
        plot(x,y,'b:'), hold on
    elseif t==2
        coeff2 = regress(AvE(1,41:inc:200,t)',X);
        y = coeff2'*[ones(size(x)); x];
        plot(log10([1:inc:200]),AvE(1,1:inc:200,t),'g.-'), hold on
        plot(x,y,'g:'), hold on
    end
end
legend('Kernel t=1',num2str(-coeff1(2)),'DPP',num2str(-coeff2(2)))
toc %+ 0.2 secs
savefig('DPPkernel2')