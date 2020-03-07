function boundrytemp = transientBCHeatedBlock(~, state)
%boundaryFileHeatedBlock Temperature boundary conditions for heated block example
% Temperature boundary condition is defined on the left edge of the block
% in the heated block example.
%
% loc   - application region struct passed in for information purposes
% state - solution state struct passed in for information purposes

% Copyright 2014-2016 The MathWorks, Inc.

% The temperature returned depends on the solution time.
global inittemp
global risetemp
global finaltemp
%initial_temp=20;
rise_time = 60*risetemp;
sim_time = 60*60;
if(isnan(state.time))
  % Returning a NaN for any component of q, g, h, r when time=NaN
  % tells the solver that the boundary conditions are functions of time.
  % The PDE Toolbox documentation discusses this requirement in more detail.
  boundrytemp = NaN;
elseif(state.time <= (rise_time)) % 3 degree / min
  % From time=0 to time=.5, the temperature ramps from zero to 100.
  boundrytemp = (finaltemp-inittemp)*state.time/(rise_time)+inittemp;
else
  % For time > .5, the temperature is fixed at 100
  boundrytemp = finaltemp;
end
end