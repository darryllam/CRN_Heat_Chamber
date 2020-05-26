close all; clear all; clc;

% https://www.mathworks.com/help/pde/ug/pde.pdemodel.importgeometry.html
% Cylinder dimensions: 1.5cm radius, 5cm length

model = createpde(3);
importGeometry(model, 'capstone_cylinder.stl');

pdegplot(model, 'FaceLabels', 'on');


