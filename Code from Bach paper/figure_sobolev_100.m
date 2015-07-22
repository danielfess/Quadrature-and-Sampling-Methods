
clear all
seed=1;
randn('state',seed);
rand('state',seed);


n = 200;

% kernels
nk = 4;
kernels{1} = @(x,y)   ( 1 + pi^2 * ( frac_dist(x,y).^2 - frac_dist(x,y) + 1/6 ) ) ;
kernels{2} = @(x,y)   ( 1 + pi^4 / 3 * ( -frac_dist(x,y).^4 + 2 *frac_dist(x,y).^3 - frac_dist(x,y).^2 + 1/30 ) ) ;
kernels{3} = @(x,y)   ( 1 + 2 * pi^6 / 45 * ( frac_dist(x,y).^6 - 3 * frac_dist(x,y).^5 + 5/2 *frac_dist(x,y).^4 - 1/2 * frac_dist(x,y).^2 + 1/42 ) ) ;
kernels{4} = @(x,y)   ( 1 + pi^8 / 315 * ( -frac_dist(x,y).^8 + 4 * frac_dist(x,y).^7 - 14/3 * frac_dist(x,y).^6 + 7/3 *frac_dist(x,y).^4 - 2/3 * frac_dist(x,y).^2 + 1/30 ) ) ;


for irep = 1:100
    irep
    X = rand(1,n);
    
    
    for j=1:nk
        Ks{j} = zeros(length(X),length(X));
        for i=1:length(X), Ks{j}(:,i) = kernels{j}(X,X(i)); end
    end
    
    for m=1:n
        
        for j=1:nk
            
            
            % alphas{j} =  ( Ks{j}(1:m,1:m)+ 1e-16 *eye(m) *m ) \ ones(m,1);
            
            
            [G,pp,ee,npp] = icd_general(Ks{j}(1:m,1:m),1e-16,m);
            G = G(:,1:ee);
            u = G(1:ee,1:ee) \ ones(ee,1);
            a = G(1:ee,1:ee)' \ u;
            alphas{j}   = zeros(m,1); alphas{j}(pp(1:ee)) = a;
            
            for k=1:nk
                errors(j,k,m,irep) = max(alphas{j}' * ( Ks{k}(1:m,1:m)*alphas{j}) - 2 * sum(alphas{j}) + 1,0);

            end
        end
    end
end
errors = mean(errors,4);

save data_sobolev
open('figure_sobolev.fig');
for i=1:nk
    subplot(2,2,i)
    set(gca,'fontsize',18)
    plot(log10(1:n),log10(squeeze(errors(:,i,:))),'linewidth',2)
    for j=1:4
        [a,b]=affine_fit(log10(n/4:n),log10(squeeze(errors(j,i,n/4:n))));
        as(i,j)=a;
        bs(i,j)=b;
        
    end
    hold on;
    plot(log10(1:n),  as(i,:)'*log10(1:n) +bs(i,:)'*ones(1,n),':','linewidth',2);
    hold off
    if i==1
        legend(sprintf('t=1 : %1.1f',-as(i,1)),sprintf('t=2 : %1.1f',-as(i,2)),sprintf('t=3 : %1.1f',-as(i,3)),sprintf('t=4 : %1.1f',-as(i,4)), 'Location','NorthEast');
    else
        legend(sprintf('t=1 : %1.1f',-as(i,1)),sprintf('t=2 : %1.1f',-as(i,2)),sprintf('t=3 : %1.1f',-as(i,3)),sprintf('t=4 : %1.1f',-as(i,4)), 'Location','SouthWest');
        
    end
    xlabel('log_{10}(n)');
    ylabel('log_{10}(squared error)');
    axis([ 0 log10(n) min(min(log10(squeeze(errors(:,i,:))))) max(0,max(max(log10(squeeze(errors(:,i,:))))))]);
    title(sprintf('Test functions : s = %d',i));
end



print('-depsc', 'figure_sobolev.eps' );
