%Example to show scattering of points
%Draw using n-DPP with SE kernel vs. Poisson point process

tic
x = rand(2,500);
lambda = 50;

%kernel = @(x,y) ( (x(1,:).*y(1,:)+x(2,:).*y(2,:)) )
kernel = @(x,y)   ( exp(-sum((x-y).^2,1)/2) );

K = zeros(length(x));
for i=1:length(x)
    K(i,:) = kernel(repmat(x(:,i),1,length(x)),x);
end

L = decompose_kernel(K);

Y1 = x(:,sample_dpp(L,lambda)); %n-DPP sample
%Technically we are drawing from a n-DPP, not a DPP, but it helps to
%control the size the figures are more comparable, and the n-DPP sample has
%the same qualities as a DPP sample would.

npoints = poissrnd(lambda);
Y2 = rand(2,npoints); %Poisson PP sample

figure(1)
scatter(Y1(1,:),Y1(2,:),50,'.')
legend('DPP')
figure(2)
scatter(Y2(1,:),Y2(2,:),50,'.')
legend('Independent')
toc
