classdef PCGSteihaugOptimizer < handle
    %PCGOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        model
        thrustradius
        log = Log('loss',...
                  'newloss',...
                  'thrustradius',...
                  'nstep',...
                  'reductionratio',...
                  'flag',...
                  'relres',...
                  'niter');
    end
    properties % options
        thrustradius_max        = Inf
        thrustradius_decrease   = 1/4
        thrustradius_increase   = 2
        rejection_threshold     = 0
        reltol                  = 1e-6
        maxiter                 
        preconditioner          = []
    end
    
    methods
        function opt = PCGSteihaugOptimizer(model, initialthrustradius, varargin)
            %PCGOPTIMIZER Construct an instance of this class
            %   Detailed explanation goes here
            opt.model = model;
            opt.thrustradius = initialthrustradius;
            
            % default options
            opt. maxiter = model.paramsize; 
            for i=1:2:numel(varargin)
                opt.(varargin{i}) = varargin{i+1};
            end
        end
        
        function reset(~) 
        end
        
        function accept = update(opt, reductionratio, maxstep)
            if reductionratio < 0.25
                opt.thrustradius = opt.thrustradius*opt.thrustradius_decrease;
            else
                if reductionratio > 0.75 && maxstep
                    opt.thrustradius = opt.thrustradius*opt.thrustradius_increase;
                    opt.thrustradius = min(opt.thrustradius, opt.thrustradius_max);
                end
            end
            accept = (reductionratio > opt.rejection_threshold);
        end
        
        function plot(opt)
%             figure(1);
%             plot(getfield(opt.log, 'loss')), hold on

%             figure(2);
            subplot(2,2,1); plot(max(getfield(opt.log, 'reductionratio'),0)), hold on
            subplot(2,2,2); plot(getfield(opt.log, 'niter')), hold on
            subplot(2,2,3); semilogy(getfield(opt.log, 'thrustradius'), '--'), hold on
                            plot(getfield(opt.log, 'nstep'))
            subplot(2,2,4); plot(getfield(opt.log, 'relres')), hold on
        end
        
        function [newloss] = step(opt, loss)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            M = opt.preconditioner;
            if ~isnumeric(M) % M is function handle
                M = M(opt.model);
            end
            
            g = bdiff(opt.model, [], 1);
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
            [step, flag, relres, iter] = cpcg(@(v) gvp(opt.model, v), -g, tol, ...
                opt.maxiter, M, [], opt.thrustradius);
            
            % compute predicted reduction along step
            [~, loss_s] = fdiff(opt.model, step);
            s_Gloss_s = dot(step, gvp(opt.model, step));
            predchange = loss_s + s_Gloss_s/2;
            
            % make step and compute reduction ratio
            opt.model.params = opt.model.params + step;
            [~, newloss] = recall(opt.model);
            reductionratio = (newloss-loss)/predchange;
            
            tr = opt.thrustradius;
            % take back step if not accepted
            if ~update(opt, reductionratio, flag==5)
                opt.model.params = opt.model.params - step;
                newloss = loss;
                step = zeros(numel(step),1);
            end
                
            append(opt.log, loss, newloss, tr, norm(step), ...
                            reductionratio, flag, relres, iter);
        end
    end
end

