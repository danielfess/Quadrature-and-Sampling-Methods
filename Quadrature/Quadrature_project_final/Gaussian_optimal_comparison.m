%Comparing optimal dist. for SE kernel, Gaussian measure on R, for various
%lambda

tic
m = 100; %Truncating point
si2 = 1;
t2 = 1;
l = [1 0.1 0.01 0.001 0.0001];

x1 = -5:1/200:5;

q = zeros(m,length(x1),5);
%(i,j,t)th entry is ei(x1j)*(eval(i)/(eval(i)+l(t)))

a = 1/(4*t2);
b = 1/(2*si2);
c = sqrt(a^2+2*a*b);
A = a+b+c;
B = b/A;

eval = zeros(m,1); %Eigenvalues
for i = 0:m-1
    eval(i+1) = sqrt(2*a/A)*(B^i);
end

qsum = zeros(length(x1),5);
density = qsum;
for t=1:5
    for i = 0:m-1
        q(i+1,:,t) = (eval(i+1)/(eval(i+1)+l(t)))*exp(-2*(c-a)*(x1.^2)).*(hermiteH(i,sqrt(2*c)*x1).^2)./(sqrt(a/c)*(2^i)*factorial(i));
    end
    toc
    
    qsum(:,t) = sum(q(:,:,t),1)/sum(eval./(eval+l(t))); %qsum(i,t) is approx optimal density wrt gaussian at x1i
    density(:,t) = qsum(:,t)'.*exp(-2*a*x1.^2)*sqrt(2*a/pi);
end

figure (2)
clf
for t=1:5
plot(x1,density(:,t))
hold on
end
y = normpdf(x1,0,sqrt(t2));
plot(x1,y)
title('Gaussian optimal for various lambda')
legend(sprintf('Lambda %g',l(1)),sprintf('Lambda %g',l(2)),sprintf('Lambda %g',l(3)),sprintf('Lambda %g',l(4)),sprintf('Lambda %g',l(5)),'Measure')
%savefig('gaussian_optimal_comparison')