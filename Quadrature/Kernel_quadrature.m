%Quadrature

%Method:
% 1) Draw x1,...,xn from U[0,1]
% 2) Sobolev space parameter t, kernel K_t. Beta = K(t).1 (Kernel matrix *
% vector of ones)
% 3) Produce test function h in RKHS (here Sobolev parameter s) of norm 1,
% and use to test quadrature method:
% - Draw y1,...,ym U[0,1], m=500 - why U[0,1] not e.g. N(0,1)
% - Draw c1,...,cm N(0,1)
% - h = sum ci*K_s(.,yi) (lies in RKHS)
% - Normalise h, ||h|| = c'*K_s(y1,..,ym)*c
% - Plot h
% 4) Perform quadrature
% 5) Because here int(K_s(x,y) dx) = 1, int(h) = sum ci
% 6) Calculate squared error in quadrature
% 7) Log-log plot of averaged errors

tic
clf
v = 200; s=1; t=1;

for j=1:10
x = rand(v,1);
%Need matrix where i,j entry is xi - xj
%Idea: Matrix X1 duplicate columns then M = X1 - X1'
l=1; x1 = zeros(v);
while l<=v, x1(:,l) = x; l=l+1;
end
K_t = 1 + (((-1)^(t-1))*((2*pi)^(2*t))/(2*factorial(2*t)))*bernoulli(2*t,abs(x1'-x1)-floor(abs(x1'-x1)));
E = zeros(10,200);

%h = sum ci*h(.,yi), and normalise h
r=100; %Normally 500
y = rand(r,1);
c = randn(r,1);
l=1; y1 = zeros(r);
while l<=r, y1(:,l) = y; l=l+1;
end
K_s = 1 + (((-1)^(s-1))*((2*pi)^(2*s))/(2*factorial(2*s)))*bernoulli(2*s,abs(y1'-y1)-floor(abs(y1'-y1)));
c = c/sqrt(c'*K_s*c);

z = [0:0.01:1];
m=1; h = zeros(r,length(z));
while m<=length(z), h(:,m) = c.*(1 + (((-1)^(s-1))*((2*pi)^(2*s))/(2*factorial(2*s)))*bernoulli(2*s,abs(z(m)*ones(r,1)-y)-floor(abs(z(m)*ones(r,1)-y)))); m=m+1;
end
h = sum(h,1);
figure(2)
plot(z,h)

for n = 1:20:v
    K = K_t(1:n,1:n);

    b = K\ones(n,1);
%Try (K_t)\1 and recreate the first plot

m=1; h = zeros(r,n);
while m<=n, h(:,m) = c.*(1 + (((-1)^(s-1))*((2*pi)^(2*s))/(2*factorial(2*s)))*bernoulli(2,abs(x(m)*ones(r,1)-y)-floor(abs(x(m)*ones(r,1)-y)))); m=m+1;
end
h = sum(h,1);

A = h*b;
I = sum(c);
E2 = (A-I)^2;
E(j,n) = E2;
end
E = sum(E,1)/max(j)
end

figure(1)
plot(log([1:20:v]),log(E(1:20:v)))

toc