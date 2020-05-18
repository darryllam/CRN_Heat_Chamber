close all;
clc;

tic

initTempRange = 15;
finalTempRange = 40:5:60;
riseTempRange = 15;
specificHeatRange = [921, 385, 460];%, 380];
thermalConductivityRange = [150, 388, 79.5];
massDensityRange = [2.70*1000, 8.933*1000, 7.849*1000];%, 8.8*1000];
materialRange = ["Al", "Cu",  "Fe"];% "Brass"];
% initTempRange = 15;
% finalTempRange = 60;
% riseTempRange = 15;
% specificHeatRange = [921];
% thermalConductivityRange = [150];
% massDensityRange = [2.70*1000];
% materialRange = ["Al"];
rad_array = [1.75, 2.25, 2.75];
mat_iteration_array = cell(length(specificHeatRange), length(thermalConductivityRange), length(massDensityRange), length(materialRange));
mat_iterations = size(mat_iteration_array);
temprad_iteration_array = cell(length(initTempRange), length(finalTempRange), length(riseTempRange), length(rad_array));
temprad_iterations = size(temprad_iteration_array);
delete(gcp('nocreate'))
parpool('local',10);

for i = 1:prod(temprad_iterations)
    
    thermalmodelT = createpde('thermal','transient');
    [initTempIter, finalTempIter, riseTempIter, radIter] = ind2sub(temprad_iterations, i);
    rad = rad_array(radIter);
    inittemp = initTempRange(initTempIter);
    risetemp = riseTempRange(riseTempIter);
    finaltemp = finalTempRange(finalTempIter);
    air = rad + 0.05;
    %define the shape of the 'air'
    r1 = [-air air air -air -air -air air air];
    r1 = [3 4 r1./100]';
    %define the shape of the 2d sqaure representing a cylinder
    r2 = [-rad rad rad -rad  -rad -rad rad rad];
    r2 = [3 4 r2./100]';
    gm = [r1,r2];
    sf = 'r1+r2'; %add the shapes together
    ns = char('r1','r2');
    ns = ns';
    [dl,bt] = decsg(gm,sf,ns);
    geometryFromEdges(thermalmodelT,dl); %thermal model is now a
    
    k_dc = finaltemp-inittemp;
    overshoot = 13;
    settling_time = 200;
    zeta = sqrt(log(overshoot/100)^2/(pi^2 + log(overshoot/100)^2));
    w_n = 4/(settling_time*zeta);
    [~,den] = ord2(w_n, zeta);
    num = k_dc*(w_n)^2;
    sys = tf(num,den);
    %         [A,B,C,D] = tf2ss(num,den);
    %         sys = ss(A,B,C,D);
    %         [yi,ti,xi] = initial(sys,[inittemp,0],60*60);
    %         sys = tf(num,den);
    [ys,~,~] = step(sys);
    airtemp = inittemp + ys;
    
    parfor (j = 1:prod(mat_iterations))
%     for (i = 1:prod(iterations))
        [specHeatIter, thermCondIter, massDensIter, matIter] = ind2sub(mat_iterations,j);
        

        thermalConductivity = thermalConductivityRange(thermCondIter);
        massDensity = massDensityRange(massDensIter);
        specificHeat = specificHeatRange(specHeatIter);
        material = materialRange(matIter);
        
        thermalProperties(thermalmodelT,'Face',1,'ThermalConductivity',25.72/1000,... %W/m K
            'MassDensity',1.2041 ,... %kg/m^3
            'SpecificHeat', 0.718/1000); %J/(kmol*K)
        thermalProperties(thermalmodelT,'Face',2,'ThermalConductivity',thermalConductivity,... %W/m K
            'MassDensity',massDensity,... %kg/m^3
            'SpecificHeat',specificHeat); %J/(kg k)
        
        transientBCHeatedBlockParallelPID(thermalmodelT, 'Edge', [2,1,7,6], 'Temperature', inittemp, airtemp, finaltemp)

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
        specificHeatCSV = zeros(length(tlist),1) + specificHeat;
        thermalConductivityCSV = zeros(length(tlist),1) + thermalConductivity;
        massDensityCSV = zeros(length(tlist),1) + massDensity;
        radCSV = zeros(length(tlist),1) + rad;
        
        csvName = sprintf("%.2f%sTemps%d%d%dPID.csv", rad, material, inittemp,risetemp,finaltemp)
        csvMat = [tlist', outT', centerT', radCSV, radCSV, specificHeatCSV, thermalConductivityCSV, massDensityCSV];
        writematrix(csvMat,csvName);
 
        %     save(Namecenter,'centerT')
        %     save(Nameouter, 'outT')
    end
    
end
toc