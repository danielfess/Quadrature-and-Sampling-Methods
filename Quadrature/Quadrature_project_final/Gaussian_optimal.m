%Approximate optimal dist. for SE kernel + gaussian measure on R

tic
m = 100; %Truncating point of sum
%Input si2, t2, lambda from other script, else:
if exist('si2','var')==0
    si2 = 1;
end
if exist('t2','var')==0
    t2 = 1;
end
if exist('lambda','var')==0
    lambda = 0.0001;
end

x1 = -5:1/200:5;

q = zeros(m,length(x1));
%(i,j)th entry is ei(x1j)*(eval(i)/(eval(i)+lambda))

a = 1/(4*t2);
b = 1/(2*si2);
c = sqrt(a^2+2*a*b);
A = a+b+c;
B = b/A;

eval = zeros(m,1); %Eigenvalues
for i = 0:m-1
    eval(i+1) = sqrt(2*a/A)*(B^i);
end

for i = 0:m-1
    q(i+1,:) = (eval(i+1)/(eval(i+1)+lambda))*exp(-2*(c-a)*(x1.^2)).*(hermiteH(i,sqrt(2*c)*x1).^2)./(sqrt(a/c)*(2^i)*factorial(i));
    %Note the normalisation constant in the eigenvector.
end
toc

qsum = sum(q,1)/sum(eval./(eval+lambda)); %qsum(i) is approx optimal density wrt gaussian at x1i.
density = qsum.*exp(-2*a*x1.^2)*sqrt(2*a/pi); %Approx density wrt lebesgue measure.

figure (2)
clf
plot(x1,density)
sigma = 1/(sqrt(2*pi)*density(1001));
y = normpdf(x1,0,sigma);
hold on
plot(x1,y)
y = normpdf(x1,0,sqrt(t2));
hold on
plot(x1,y)
legend('Optimal','Best fit Normal','Measure')