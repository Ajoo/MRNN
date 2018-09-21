classdef PCGStep < handle
    %ADAMOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties % algorithm options
        preconditioner          = [];
        reltol                  = 1e-6;
        maxiter                 
        
        log_pcg                 = Log('nstep',...
                                      'flag',...
                                      'relres',...
                                      'niter');
    end
    
    methods (Sealed)
        function reset(opt)
        end
        
        function [step, predchange, limited, gnstep] = compute_step(opt, maxstep)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            g = grad(opt);
            
            M = opt.preconditioner;
            if ~isnumeric(M) % M is function handle
                M = M(opt.model);
            end
            
            if isnumeric(opt.reltol)
                tol = opt.reltol;
            else
                if ~isempty(M)
                    ng = sqrt(g'*(M\g));
                else
                    ng = norm(g);
                end
                tol = opt.reltol(ng);
            end

            % compute step
            if nargin >= 2
                [step, flag, relres, iter] = cpcg(@opt.gvp, -grad(opt), tol, ...
                    opt.maxiter, M, [], maxstep);
                limited = flag==5;
            else
                [step, flag, relres, iter] = cpcg(@opt.gvp, -grad(opt), tol, ...
                    opt.maxiter, M, [], inf);
                limited = false;
            end
            
            if nargout >= 2
                % compute predicted reduction along step
                [~, loss_s] = fdiff(opt.model, step);
                [~, s_Gloss_s] = gvp(opt.model, step);
                % s_Gloss_s = dot(step, gvp(opt.model, step));
                predchange = loss_s + s_Gloss_s/2;
                gnstep = sqrt(s_Gloss_s);
            end
            
            append(opt.log_pcg, norm(step), flag, relres, iter);
        end
    end
end