function [x,flag,relres,iter,resvec] = cpcg(A,b,tol,maxit,M1,M2,maxnorm,varargin)
%CPCG   Constrained Preconditioned Conjugate Gradients Method.
%   X = PCG(A,B) attempts to solve the system of linear equations A*X=B for
%   X. The N-by-N coefficient matrix A must be symmetric and positive
%   definite and the right hand side column vector B must have length N.
%
%   X = PCG(AFUN,B) accepts a function handle AFUN instead of the matrix A.
%   AFUN(X) accepts a vector input X and returns the matrix-vector product
%   A*X. In all of the following syntaxes, you can replace A by AFUN.
%
%   X = PCG(A,B,TOL) specifies the tolerance of the method. If TOL is []
%   then PCG uses the default, 1e-6.
%
%   X = PCG(A,B,TOL,MAXIT) specifies the maximum number of iterations. If
%   MAXIT is [] then PCG uses the default, min(N,20).
%
%   X = PCG(A,B,TOL,MAXIT,M) and X = PCG(A,B,TOL,MAXIT,M1,M2) use symmetric
%   positive definite preconditioner M or M=M1*M2 and effectively solve the
%   system inv(M)*A*X = inv(M)*B for X. If M is [] then a preconditioner
%   is not applied. M may be a function handle MFUN returning M\X.
%
%   X = PCG(A,B,TOL,MAXIT,M1,M2,X0) specifies the initial guess. If X0 is
%   [] then PCG uses the default, an all zero vector.
%
%   [X,FLAG] = PCG(A,B,...) also returns a convergence FLAG:
%    0 PCG converged to the desired tolerance TOL within MAXIT iterations
%    1 PCG iterated MAXIT times but did not converge.
%    2 preconditioner M was ill-conditioned.
%    3 PCG stagnated (two consecutive iterates were the same).
%    4 one of the scalar quantities calculated during PCG became too
%      small or too large to continue computing.
%
%   [X,FLAG,RELRES] = PCG(A,B,...) also returns the relative residual
%   NORM(B-A*X)/NORM(B). If FLAG is 0, then RELRES <= TOL.
%
%   [X,FLAG,RELRES,ITER] = PCG(A,B,...) also returns the iteration number
%   at which X was computed: 0 <= ITER <= MAXIT.
%
%   [X,FLAG,RELRES,ITER,RESVEC] = PCG(A,B,...) also returns a vector of the
%   estimated residual norms at each iteration including NORM(B-A*X0).
%
%   Example:
%      n1 = 21; A = gallery('moler',n1);  b1 = A*ones(n1,1);
%      tol = 1e-6;  maxit = 15;  M = diag([10:-1:1 1 1:10]);
%      [x1,flag1,rr1,iter1,rv1] = pcg(A,b1,tol,maxit,M);
%   Or use this parameterized matrix-vector product function:
%      afun = @(x,n)gallery('moler',n)*x;
%      n2 = 21; b2 = afun(ones(n2,1),n2);
%      [x2,flag2,rr2,iter2,rv2] = pcg(@(x)afun(x,n2),b2,tol,maxit,M);
%
%   Class support for inputs A,B,M1,M2,X0 and the output of AFUN:
%      float: double
%
%   See also BICG, BICGSTAB, BICGSTABL, CGS, GMRES, LSQR, MINRES, QMR,
%   SYMMLQ, TFQMR, ICHOL, FUNCTION_HANDLE.

%   Copyright 1984-2013 The MathWorks, Inc.

if (nargin < 2)
    error(message('MATLAB:cpcg:NotEnoughInputs'));
end

m = size(b,1);
n = m;
if ~iscolumn(b)
    error(message('MATLAB:cpcg:RSHnotColumn'));
end

% Assign default values to unspecified parameters
if (nargin < 3) || isempty(tol)
    tol = 1e-6;
end
if tol <= eps
    warning(message('MATLAB:pcg:tooSmallTolerance'));
    tol = eps;
elseif tol >= 1
    warning(message('MATLAB:pcg:tooBigTolerance'));
    tol = 1-eps;
end
if (nargin < 4) || isempty(maxit)
    maxit = min(n,20);
end

existM1 = ((nargin >= 5) && ~isempty(M1));
existM2 = ((nargin >= 6) && ~isempty(M2));

if nargin < 7 || isempty(maxnorm)
    maxnorm2 = Inf;
else
    maxnorm2 = maxnorm^2;
end

% Set up for the method
x = zeros(n,1);
flag = 1;
r = b;
% In "square root" form compute rp = E\r; zr = <rp, rp>
if existM1
    z = M1\r;
else
    z = r;
end
if existM2
    z = M2\z;
end
rz = r'*z;
normr = sqrt(rz);
tolrel = tol * normr;              % Relative tolerance

if (normr <= tolrel)                 % Initial guess is a good enough solution
    flag = 0;
    relres = 1;
    iter = 0;
    resvec = tolrel;
    return
end

resvec = zeros(maxit+1,1);         % Preallocate vector for norm of residuals
resvec(1,:) = normr;                  % resvec(1) = norm(b-A*x0)

% loop over maxit iterations (unless convergence or failure)
for ii = 1:maxit
    [q, pq] = A(p,varargin{:});
    % if pq < 0 negative curvature was found
    alpha = rz/pq;
    x = x + alpha * p;             % form new iterate
    
    % check if x reached max norm
    if existM2
        y = M2*x;
    else
        y = x;
    end
    if existM1
        y = M1*y;
    end
    xy = x'*y;
    overstep = xy - maxnorm2;
    if overstep > 0
        flag = 5;
        py = p'*y;
        if existM2
            d = M2*p;
        else
            d = p;
        end
        if existM1
            d = M1*d;
        end
        pd = p'*d;
        tau = (py - sqrt(py^2 - overstep*pd))/pd;
        break
    end
    
    if existM1
        y = M1\r;
        if ~all(isfinite(y))
            flag = 2;
            break
        end
    else % no preconditioner
        y = r;
    end
    
    if existM2
        z = M2\y;
        if ~all(isfinite(z))
            flag = 2;
            break
        end
    else % no preconditioner
        z = y;
    end
    
    rho1 = rho;
    rho = r' * z;
    if ((rho == 0) || isinf(rho))
        flag = 4;
        break
    end
    if (ii == 1)
        p = z;
    else
        beta = rho / rho1;
        if ((beta == 0) || isinf(beta))
            flag = 4;
            break
        end
        p = z + beta * p;
    end
    q = A(p,varargin{:});
    pq = p' * q;
    alpha = rho / pq;
    
    x = x + alpha * p;             % form new iterate
    normx = norm(x);
    if normx >= maxnorm
        flag = 0;
        return
    end
    r = r - alpha * q;
    normr = norm(r);
    normr_act = normr;
    resvec(ii+1,1) = normr;
    
    % check for convergence
    if (normr <= tolb)
        r = b - iterapp('mtimes',afun,atype,afcnstr,x,varargin{:});
        normr_act = norm(r);
        resvec(ii+1,1) = normr_act;
        flag = 0;
        iter = ii;
    end
    if (normr_act < normrmin)      % update minimal norm quantities
        normrmin = normr_act;
        xmin = x;
        imin = ii;
    end
end                                % for ii = 1 : maxit

% returned solution is first with minimal residual
relres = normr_act / n2b;

% truncate the zeros from resvec
if ((flag <= 1) || (flag == 3))
    resvec = resvec(1:ii+1,:);
else
    resvec = resvec(1:ii,:);
end
