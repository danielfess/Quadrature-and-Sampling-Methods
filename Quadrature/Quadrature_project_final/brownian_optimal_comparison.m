%Comparing optimal dist. for brownian kernel + uniform measure on [0,1] for
%different lambda

tic
figure(2);clf;
m = 1000; %Truncating point
lambda = [1 0.1 0.01 0.001];

x1 = 0:1/10000:1;


q = zeros(m,length(x1),4);
%(i,j)th entry is ei(xj)*(eval(i)/(eval(i)+lambda))

eval = zeros(m,1); %Eigenvalues
for i = 0:m-1
    eval(i+1) = 1/((i+0.5)*pi)^2;
end

for t=1:4
    for i = 0:m-1
        q(i+1,:,t) = (eval(i+1)/(eval(i+1)+lambda(t)))*2*(sin((i+0.5)*pi*x1).^2);
    end
    toc
    
    q = sum(q,1)/sum(eval./(eval+lambda(t))); %approx optimal density
    
    plot(x1,q(1,:,t)); hold on
end
title('Optimal Density for Brownian 1D with U[0,1] measure')
legend('1','0.1','0.01','0.001')
savefig('brownian_optimal_comparison')