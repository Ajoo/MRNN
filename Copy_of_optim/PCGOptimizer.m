classdef PCGOptimizer < handle
    %PCGOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        model
        options
        state
        log
    end
    
    methods
        function opt = PCGOptimizer(model, initialdamping, varargin)
            %PCGOPTIMIZER Construct an instance of this class
            %   Detailed explanation goes here
            opt.model = model;
            opt.state = struct('damping', initialdamping,...
                               'previousstep', zeros(model.paramsize, 1));
            
            % default options
            opt.options = struct('MaxDamping', Inf, ...
                                 'RejectionThreshold', 0, ...
                                 'RelTol', 1e-6, ...
                                 'MaxIter', model.paramsize, ...
                                 'Preconditioner', [], ...
                                 'PreconditionerSupression', 1,...
                                 'HotRestart', true);
            
            for i=1:2:numel(varargin)
                opt.options.(varargin{i}) = varargin{i+1};
            end       
        end
        
        function accept = update(opt, reductionratio)
            damping = opt.state.damping;
            if reductionratio < 0.25
                damping = damping*3/2;
            elseif reductionratio > 0.75
                damping = damping*2/3;
            end
            opt.state.damping = damping;
            accept = (reductionratio > opt.options.RejectionThreshold);
        end
        
        function [newloss, reductionratio, flag, relres, iter] = step(opt, loss)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            g = bdiff(opt.model, [], 1);
            
            damping = opt.state.damping;
            preconditioner = opt.options.Preconditioner;
            if ~isempty(preconditioner)
                alpha = opt.options.PreconditionerSupression;
                minv = (preconditioner(opt.model) + damping).^-alpha;
                preconditioner = @(x) minv.*x;
            end
            
            
            if isnumeric(opt.options.RelTol)
                tol = opt.options.RelTol;
            else
                if ~isempty(preconditioner)
                    ng = sqrt(g'*preconditioner(g));
                else
                    ng = norm(g);
                end
                tol = opt.options.RelTol(ng);
            end
            
            
            if opt.options.HotRestart
                x0 = opt.state.previousstep;
            else
                x0 = [];
            end
            % compute step
            [step, flag, relres, iter] = pcg(@gvp_, -g, tol, opt.options.MaxIter, ...
                preconditioner, [], x0);
            
            % compute predicted reduction along step
            [~, loss_s] = fdiff(opt.model, step);
            s_Gloss_s = dot(step, gvp(opt.model, step));
            predchange = loss_s + s_Gloss_s/2;
            
            % make step and compute reduction ratio
            opt.model.params = opt.model.params + step;
            [~, newloss] = recall(opt.model);
            reductionratio = (newloss-loss)/predchange;
            
            % take back step if not accepted
            if ~update(opt, reductionratio)
                opt.model.params = opt.model.params - step;
                newloss = loss;
                opt.state.previousstep = zeros(numel(step),1);
            else
                opt.state.previousstep = step;
            end
                
            function gv = gvp_(v)
                gv = gvp(opt.model, v) + damping*v;
            end
        end
    end
end

