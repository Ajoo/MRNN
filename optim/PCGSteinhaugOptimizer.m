classdef PCGSteinhaugOptimizer < PCGStep & DeterministicTROptimizer
    %ADAMOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    methods
        function opt = PCGSteinhaugOptimizer(varargin)
            opt@PCGStep();
            opt@DeterministicTROptimizer(varargin{:});
        end
    end
end