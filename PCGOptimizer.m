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
                                 'MaxIter', model.paramsize);
            
            for i=1:numel(varargin)/2
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
        
        function [reductionratio, flag, relres, iter] = step(opt, loss)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            g = bdiff(opt.model, [], 1);
            ng = norm(g);
            if isnumeric(opt.options.RelTol)
                tol = opt.options.RelTol;
            else
                tol = opt.options.RelTol(ng);
            end
            
            damping = opt.state.damping;
            % compute step
            [step, flag, relres, iter] = pcg(@gvp_, -g, tol, opt.options.MaxIter);
            
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
            end
            
            opt.state.previousstep = step;
            function gv = gvp_(v)
                gv = gvp(opt.model, v) + damping*v;
            end
        end
    end
end

