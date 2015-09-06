%Functions from the RKHS with brownian covariance kernel
%We will only give examples of functions on [0,1] since this is what we
%integrate over in our quadrature examples.

kernel = @(x,y) ( min(x,y) )
figure(1);clf

for t=1:4 %4 example functions

%Our function will look like sum[ c_i*k(.,y_i) ]
r=1000; %No. terms
y = rand(1,r); %for k(.,y_i)
c = randn(1,r); %coeffs for k(.,y_i)
K_y = kernel(repmat(y,r,1)',repmat(y,r,1)); %Kernel matrix for y
c = c/sqrt(c*K_y*c'); %Normalise wrt RKHS norm

x = 0:0.0001:1; %We will evaluate function on x.
h = zeros(r,length(x));
%(i,j)-th entry will be c_i*k(x_j,y_i) then we will sum over i
for i = 1:r
    h(i,:) = c(i)*kernel(x,y(i));
end
h = sum(h,1); %Summing the terms
subplot(2,2,t)
plot(x,h)
title(sprintf('Function %g',t))
end
