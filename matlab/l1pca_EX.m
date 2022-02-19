%L1PCA_EX Exhaustive-search, exact L1-principal component analysis.
%  [Qopt, Bopt]=L1PCA_EX(X, K) produces a unitary matrix Qopt, with 
%  as many rows as X (say, D) and K columns (D>=K), and a binary
%  matrix Bopt (each entry is either +1  or -1) of as many rows as
%  the columns of X (say, N) and K columns, so that 
%  (i) Qopt=argmax_{Q'*Q=eye(K)} sum(sum(abs(X'*Q))) and
%  (ii) Bopt=argmax_{each entry of B is either +1 or -1} sum(svd((X*B)).
%  The columns of Qopt are the K L1-principal components of X.
%  This function implements the exhaustive-search exact algorithm,
%  presented in the articles referenced below.
%
%   Author:
%   Dr. Panos P. Markopoulos
%   Electrical Engineering Department
%   Rochster Institute of Technology
%   Email: pxmeee@rit.edu
%   Web: https://people.rit.edu/pxmeee/
%
%   Created: June 2013
%
%   References: 
%   1) P. P. Markopoulos, G. N. Karystinos, and D. A. Pados,"Optimal
%   algorithms for L1-subspace signal processing," IEEE Transactions on
%   Signal Processing, vol. 62, pp. 5046-5058, Oct. 2014. 
%   2) P. P. Markopoulos, G. N. Karystinos, and D. A. Pados,"Some options
%   for L1-subspace signal processing," in Proc. 10th International
%   Symposium on Wireless Communication Systems (ISWCS 2013), 
%   Ilmenau, Germany, Aug. 2013, pp. 622-626.
%   
%   **Inquiries regarding the script provided below are cordially welcome. 
%   In case you spot a bug, please let me know. 
%   If you use some piece of code for your own work, please cite the 
%   articles above.** 

 
function [Qopt, Bopt]=l1pca_EX(X, K)
tic;
toler=1e-8;

if norm(imag(X),2)>toler
    error(['X must be a real matrix (here, norm(imag(X),2)=' num2str(norm(imag(X),2)), ').']')
end
 
[~, N]=size(X);
[~, S, ~]=svd(X);
s=diag(S);
d=sum(s>toler);

if K>d
    error(['K must be less or equal to d=rank(X) (here, d=', int2str(d), ', K=', int2str(K), ').']);
end

% num_of_cands_exhaust=nchoosek(2^(N-1)+K-1,K); 
% w=zeros(1,d);
% for i=0:d-1
%     w(i+1)=nchoosek(N-1,i);
% end
% num_of_cands_poly=nchoosek(sum(w)+K-1,K);

 
Bc=(de2bi(0:(2^(N-1)-1),N)*2-1)';

[~, a]=size(Bc);

muts=multisets(1:a,K);
[n_muts, ~]=size(muts);

mopt=-inf;
for i=1:n_muts
    
    comb=muts(i,:);
    B=Bc(:,comb);
    
    m=sum(svd(X*B,'econ'));
    if m>mopt
        mopt=m;
        Bopt=B;
    end
end
[U, ~, V]=svd(X*Bopt);
Qopt=U(:,1:K)*V';
timelapse=toc;
maxmetric=mopt;
 
%disp('------------------------------');
%disp(['Number of cadidates checked: ' int2str(n_muts)]);
%disp(['Time elapsed (sec): ' num2str(timelapse)]);
%disp(['Metric value: ' num2str(maxmetric)]);
%disp('------------------------------');




function M=multisets(S,K)
if min(size(S))>1
     error('S must be a N by 1 array.');
end
M=S(mulst(length(S),K));
if K<2
    M=M';
end

function x=mulst(n,k)
if k>1
    x=zeros(nchoosek(n+k-1,k),k);
    cnt=1;
    for i=1:n
        tmp=nchoosek(n-i+1+k-1-1,k-1);
        x(cnt:cnt-1+tmp,:)=[repmat(i,tmp,1) mulst(n-i+1,k-1)+i-1];
        cnt=cnt+tmp;
    end
elseif k==1
    x=(1:n)';
end
