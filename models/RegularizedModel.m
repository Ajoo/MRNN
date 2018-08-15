classdef RegularizedModel < handle
    %REGULARIZEDMODEL Summary of this class goes here
    %   Detailed explanation goes here
    
    % TODO: generalize this model with:
    %        - diagonal Gamma matrix
    %        - other regularization functions
    properties
        model
        gamma % regularization parameter
        pparams
    end
    
    properties (Dependent)
        params
        paramsize
    end
    
    methods
        function reg = RegularizedModel(mdl, gamma)
            %REGULARIZEDMODEL Construct an instance of this class
            %   Detailed explanation goes here
            reg.model = mdl;
            reg.gamma = gamma;
            
            update(reg);
        end
        
        function update(reg)
            reg.pparams = reg.params;
        end

        function r = dparams(reg)
            r = reg.params - reg.pparams;
        end
        
        function [yh, l] = evaluate(reg, x, y)
            [yh, l] = evaluate(reg.model, x, y);
            l = l + reg.gamma/2*sum(dparams(reg).^2);
        end
        
        function [yh, l] = call(reg, x, y)
            [yh, l] = call(reg.model, x, y);
            l = l + reg.gamma/2*sum(dparams(reg).^2);
        end
        
        function [yh, l] = recall(reg)
            [yh, l] = recall(reg.model);
            l = l + reg.gamma/2*sum(dparams(reg).^2);
        end
        
        function u_yh = bdiff(reg, u, ul)
            if nargin < 3
                ul = [];
            end
            u_yh = bdiff(reg.model, u, ul) + reg.gamma*dparams(reg);
        end
        
        function [yh_v, l_v] = fdiff(reg, v)
            [yh_v, l_v] = fdiff(reg.model, v);
            l_v = l_v + reg.gamma*dparams(reg)'*v;
        end
        
        function [gv, n2jv] = gvp(reg, v)
            [gv, n2jv] = gvp(reg.model, v);
            gv = gv + reg.gamma*v;
            n2jv = n2jv + reg.gamma*(v'*v);
        end
        
        function np = get.paramsize(reg)
            np = reg.model.paramsize;
        end
        function set.params(reg, p)
            reg.model.params = p;
        end
        function p = get.params(reg)
            p = reg.model.params;
        end
    end
end

