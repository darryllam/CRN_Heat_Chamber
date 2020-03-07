close all;
clc;

tic

init_temp_range = 15:5:25;
final_temp_range = 80:5:100;
rise_temp_range = 10:5:40;

iteration_array = cell(length(init_temp_range), length(final_temp_range), length(rise_temp_range));
iterations = size(iteration_array);

thermalmodelT = createpde('thermal','transient');

%define the shape of the 'air'
r1 = [-1.55 1.55 1.55 -1.55 -1.55 -1.55 1.55 1.55];
r1 = [3 4 r1./100]';
%define the shape of the 2d sqaure representing a cylinder
r2 = [-1.5 1.5 1.5 -1.5  -1.5 -1.5 1.5 1.5];
r2 = [3 4 r2./100]';
gm = [r1,r2];
sf = 'r1+r2'; %add the shapes together
ns = char('r1','r2');
ns = ns';
[dl,bt] = decsg(gm,sf,ns);
geometryFromEdges(thermalmodelT,dl); %thermal model is now a

parfor (i = 1:prod(iterations))
    
    [init_temp_iter, final_temp_iter, rise_temp_iter] = ind2sub(iterations,i);
    inittemp = init_temp_range(init_temp_iter);
    risetemp = rise_temp_range(rise_temp_iter);
    finaltemp = final_temp_range(final_temp_iter);
    
    thermalProperties(thermalmodelT,'Face',1,'ThermalConductivity',25.72/1000,... %W/m K
        'MassDensity',1.2041 ,... %kg/m^3
        'SpecificHeat', 0.718/1000); %J/(kmol*K)
    thermalProperties(thermalmodelT,'Face',2,'ThermalConductivity',150,... %W/m K
        'MassDensity',2.70*1000,... %kg/m^3
        'SpecificHeat',921); %J/(kg k)
    transientBCHeatedBlockParallel(thermalmodelT, 'Edge', [2,1,7,6], 'Temperature', inittemp, risetemp, finaltemp)
    msh= generateMesh(thermalmodelT,'Hmax',0.001);
    %             figure
    %             pdeplot(thermalmodelT);
    %             axis equal
    %             title 'Block With Finite Element Mesh Displayed'
    
    tlist = 0:1:(60*60);
    thermalIC(thermalmodelT,inittemp);
    R = solve(thermalmodelT,tlist);
    T = R.Temperature;
    
    getClosestNode = @(p,x,y) min((p(1,:) - x).^2 + (p(2,:) - y).^2);
    
    [~,nid] = getClosestNode( msh.Nodes, 0, 0 );
    [~,nid2] = getClosestNode( msh.Nodes, 1.5,1.5);
    
    centerT = T(nid,:);
    outT = T(nid2,:);
    csvName = sprintf("temps%d%d%d.csv", inittemp,risetemp,finaltemp)
    csvMat = [tlist', outT', centerT'];
    writematrix(csvMat,csvName);
    
%     save(Namecenter,'centerT')
%     save(Nameouter, 'outT')
end

toc