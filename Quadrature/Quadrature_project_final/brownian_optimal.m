%Approximate optimal dist. for brownian kernel + U[0,1] measure

tic
figure(2);clf;
m = 10000; %Truncating point
if exist('lambda','var')==0
    lambda = 0.001;
end

x1 = 0:1/10000:1;

q = zeros(m,length(x1));
%(i,j)th entry is ei(xj)*(eval(i)/(eval(i)+lambda))

eval = zeros(m,1); %Eigenvalues
for i = 0:m-1
    eval(i+1) = 1/((i+0.5)*pi)^2;
end

for i = 0:m-1
    q(i+1,:) = (eval(i+1)/(eval(i+1)+lambda))*2*(sin((i+0.5)*pi*x1).^2);
end
toc

q = sum(q,1)/sum(eval./(eval+lambda)); %approx optimal density

plot(x1,q)
title(sprintf('Optimal Density for Brownian 1D with U[0,1] measure, lambda=%g',lambda))
%savefig('brownian_optimal')