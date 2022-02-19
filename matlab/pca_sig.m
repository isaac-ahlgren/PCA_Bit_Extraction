
function [B,A,V,T,U] = pca_sig(M,period_length)
    T = seperate(M, period_length);

    V = abs(fft(T,length(T(1,:)),2));

    %S = cov(V);
    %S = (S + S')/2;

    [U,Bopt] = l1pca(V',1);

    A = U' * V';
    
    B = bit_extract(A);
end

% Main algorithm to calculate the vector for bit extraction
% A = vector of the dimensially reduced matrix V
% V = matrix of filtered frequency components of T going across the
% rows
% T = matrix of observational vectors going across the rows
function [B,A,V,T,U] = PCA(M, period_length)
    T = seperate(M, period_length);
    
    V = fft(T,length(T(1,:)),2);

    S = cov(V);                                % covariance matrix of X
    S = (S + S')/2;                            % resymmetrizing the matrix (needed cause matlab is shit)
    
    
    [P,Q] = eig(S);                            % eigen value decomposition
                                               % P = eigen vector matrix
                                               % Q = diagonal eigen value
                                               % matrix
                 
    U = P(:,end);                              % lower the dimension of P
                
    U = correct_orientation(U);                % correct the orientation of the eigen vector
                 
    A = U' * V';                               % project all vectors in V onto U
                                               % this is the vector we use 
                                               % for bit extraction
    B = bit_extract(A);
end

% Seperates a vector V into period_length sized row vectors
% for matrix X.
function X = seperate(V, period_length)
    vnum = fix(length(V)/period_length);
    X = zeros(vnum, period_length);
    
    for i = 1:vnum
        X(i,:) = V((i-1)*period_length + 1 : i*period_length);
    end
end

% Algorithm to correct the orientation of eigen vectors to be pointed in the
% direction of the all-ones vector.
% Math works like this:
% u . v = |u||v|cos(theta)
% U . all-ones = |U||all-ones|cos(theta)
% U will only be facing away from all-ones if cos(theta) is less than zero
function C = correct_orientation(U)
    total = sum(U);
    if real(total) < 0                          %change orientation
        U = U * -1;
    end
    C = U;
end

% takes PCA transformed signal and for each sample, determine whether it is
% a one or a zero
function B = bit_extract(A)
    m = median(A);
    B = zeros(length(A),1);
    for i = 1:length(A)
        if A(i) > m
            B(i) = 1;
        end
    end
end