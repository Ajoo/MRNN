classdef PCGSteinhaugOptimizer < handle
    %PCGOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        model
        options
        state
        log
    end
    
    methods
        function opt = PCGSteinhaugOptimizer(model, initialthrustradius, varargin)
            %PCGOPTIMIZER Construct an instance of this class
            %   Detailed explanation goes here
            opt.model = model;
            opt.state = struct('thrustradius', initialthrustradius,...
                               'previousstep', zeros(model.paramsize, 1));
            
            % default options
            opt.options = struct('MaxThrustRadius', Inf, ...
                                 'RejectionThreshold', 0, ...
                                 'RelTol', 1e-6, ...
                                 'MaxIter', model.paramsize, ...
                                 'Preconditioner', []);
            
            for i=1:2:numel(varargin)
                opt.options.(varargin{i}) = varargin{i+1};
            end       
        end
        
        function accept = update(opt, reductionratio, maxstep)
            
            thrustradius = opt.state.thrustradius;
            if reductionratio < 0.25
                thrustradius = thrustradius/4;
            else
                if reductionratio > 0.75 && maxstep
                    thrustradius = min(2*thrustradius, opt.options.MaxThrustRadius);
                end
            end
            opt.state.thrustradius = thrustradius;
            accept = (reductionratio > opt.options.RejectionThreshold);
        end
        
        function [newloss, reductionratio, flag, relres, iter] = step(opt, loss)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            np = opt.model.paramsize;
            thrustradius = opt.state.thrustradius;
            M = opt.options.Preconditioner;
            if ~isnumeric(M) % M is function handle
                M = M(opt.model);
            end
            
            g = bdiff(opt.model, [], 1);
            if isnumeric(opt.options.RelTol)
                tol = opt.options.RelTol;
            else
                if ~isempty(M)
                    ng = sqrt(g'*(M\g));
                else
                    ng = norm(g);
                end
                tol = opt.options.RelTol(ng);
            end
            
            % compute step
            [step, flag, relres, iter] = cpcg(@(v) gvp(opt.model, v), -g, tol, opt.options.MaxIter, ...
                M, [], thrustradius);
            
            % compute predicted reduction along step
            [~, loss_s] = fdiff(opt.model, step);
            s_Gloss_s = dot(step, gvp(opt.model, step));
            predchange = loss_s + s_Gloss_s/2;
            
            % make step and compute reduction ratio
            opt.model.params = opt.model.params + step;
            [~, newloss] = recall(opt.model);
            reductionratio = (newloss-loss)/predchange;
            
            % take back step if not accepted
            if ~update(opt, reductionratio, flag==5)
                opt.model.params = opt.model.params - step;
                newloss = loss;
                opt.state.previousstep = zeros(numel(step),1);
            else
                opt.state.previousstep = step;
            end
                
            function gv = gvp_(v)
                gv = gvp(opt.model, v);
            end
        end
    end
end

